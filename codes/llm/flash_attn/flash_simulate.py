import torch
import torch.nn as nn
from einops import rearrange

vocab_size = 64
n_embd = 512
emb = nn.Embedding(vocab_size, n_embd)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = n_head
        self.d_k = n_embd // config.n_head
        self.scale = self.d_k ** -0.5

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        attn_output = self.resid_dropout(self.out_proj(attn_output))

        return attn_output


class SimpleFlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout_p = config.dropout
        self.causal = True  # assuming causal for GPT-like model
        self.block_size = config.block_size  # size of blocks for tiling, now configurable

    def forward(self, x):
        b, t, c = x.size()
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [rearrange(x, 'b t h d -> (b h) t d') for x in (q, k, v)]

        output = torch.zeros_like(q)

        for i in range(0, t, self.block_size):
            i_end = min(i + self.block_size, t)
            q_block = q[:, i:i_end]

            m = torch.full((q.shape[0], i_end - i), float('-inf'), device=q.device)
            l = torch.zeros((q.shape[0], i_end - i), device=q.device)

            for j in range(0, t, self.block_size):
                j_end = min(j + self.block_size, t)
                k_block = k[:, j:j_end]
                v_block = v[:, j:j_end]

                attn_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale

                if self.causal and j > i:
                    attn_block.fill_(float('-inf'))
                elif self.causal:
                    causal_mask = torch.triu(
                        torch.ones(i_end - i, j_end - j, dtype=torch.bool, device=attn_block.device),
                        diagonal=j - i + 1)
                    attn_block.masked_fill_(causal_mask, float('-inf'))

                m_new = torch.maximum(m, attn_block.max(dim=-1)[0])
                exp_attn = torch.exp(attn_block - m_new.unsqueeze(-1))

                l_new = l * torch.exp(m - m_new) + exp_attn.sum(dim=-1)
                output_block = torch.matmul(exp_attn, v_block)

                output[:, i:i_end] += (output_block - output[:, i:i_end]) * (l / l_new).unsqueeze(-1)

                m, l = m_new, l_new

            output[:, i:i_end] /= l.unsqueeze(-1)

        output = rearrange(output, '(b h) t d -> b t (h d)', h=self.n_head)
        return self.proj(output)
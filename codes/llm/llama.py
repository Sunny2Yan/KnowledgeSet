import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 计算位置编码的频率并加入缓存 $1 / 10000^(2k / d)$
        inv_freq = 1.0 / (base ** (torch.arange(
            0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings

        # 构建位置编码矩阵
        t = torch.arange(self.max_seq_len_cached,
                         device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # 向量外积
        emb = torch.cat((freqs, freqs), dim=-1)  # 矩阵横向拼接

        # 缓存位置编码矩阵的cos和sin值，并对齐维度 [1, 1, i, 2j]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :],
                             persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :],
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 超出长度会被截断
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings

        assert self.head_dim * self.num_heads == self.hidden_size

        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """将张量重塑为 [batch_size, num_attention_heads, seq_len, head_dim]。
        """
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def rotate_half(x):
        """将输入张量的一半隐藏维度进行旋转 (最后一个维度的左右调换).
        """
        x1 = x[..., : x.shape[-1] // 2]  # 最后一个维度的左半部分
        x2 = x[..., x.shape[-1] // 2:]  # 最后一个维度的右半部分
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        # sin和cos的前两个维度总是1, 可以进行压缩.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # [1, 1, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, num_heads, seq_len, head_dim]

        # 如果有过去存储的键值对，则进行拼接
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 如果使用缓存，则存在过去的键值对，此时更新它
        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)
                                    ) / math.sqrt(self.head_dim)
        assert attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len)

        # 如果有注意力掩码，则进行掩码处理
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))

        # 计算注意力输出 (注意 fp32)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        assert attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # 如果不需要输出注意力权重，则将其设为None
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = eps

    def _norm(self, x):
        r"""$W * \frac{x}{\sqrt{\frac{1}{n} \sum_i^n{x_i^2} + \epsilon}}$"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states):
        hidden_states = self._norm(hidden_states.float()).type_as(hidden_states)

        return self.weight * hidden_states


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
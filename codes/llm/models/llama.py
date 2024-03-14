import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from dataclasses import dataclass

""""Notes:
1. llama mlp: $l_2[act(l_1(x)) * l_1(x)]$, 其中,l_1(4096-11008); l_2(11008-4096)
2. RSMNorm: $1 / (sqrt(1/n sum(x_i^2)) + eps)$
3. RoPE: freq = 1 / (10000^(2k / d)) -> sin(m * theta); cos(m * theta)
4. model: 
                                  |--------> position_ids
input_ids[batch_size, seq_length] -(embed)-> hidden_states -[decoder * 32]-> hidden_states -[norm][linear]-> logits(hidden_states, vacab_size)
                                  |--------> mask_att
5. decoder:
              |---[residual]---|                |-[residual]-|
hidden_states -[norm][self_att]-> hidden_states -[norm][mlp]-> hidden_states
6. attention
mask_att --------------------------------------------------|
             |-[linear]-> v -------------------------------|
hidden_states -[linear]-> q -|               |-> q_states -> [stt_score] -[linear]-> att_output
             |-[linear]-> k -[rotary_pos_emb]--> k_states -|
position_ids ----------------|             
"""


@dataclass
class LlamaConfig:
    dim: int = 512
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008  # MLP隐藏层size
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    norm_eps: float = 1e-6

    initializer_range: float = 0.02
    max_position_embeddings: int = 2048

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048,
                 base=10000, device=None):
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


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = cfg.max_position_embeddings

        assert self.head_dim * self.num_heads == self.hidden_size

        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
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
        self.act_fn = nn.SiLU()  # x * sigmoid(x)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = eps

    def _norm(self, x):
        r"""$W * \frac{x}{\sqrt{\frac{1}{n} \sum_i^n{x_i^2} + \epsilon}}$"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, hidden_states):
        hidden_states = self._norm(hidden_states.float()).type_as(hidden_states)

        return self.weight * hidden_states


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size
        self.num_attention_heads = cfg.num_attention_heads
        self.self_attn = Attention(cfg)
        self.mlp = MLP(hidden_size=self.hidden_size,
                       intermediate_size=self.intermediate_size, )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=cfg.norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size,
                                                eps=cfg.norm_eps)

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
            hidden_states: (batch, seq_len, embed_dim)
            attention_mask: (batch, 1, tgt_len, src_len)
            output_attentions (`bool`, *optional*): 是否返回所有attention层的参数
            use_cache (`bool`, *optional*): 是否使用past_key_values缓存键值对
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
            缓存过去的键和值投影状态。
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

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size
        self.num_hidden_layers = cfg.num_hidden_layers
        self.padding_idx = 0
        self.initializer_range = cfg.initializer_range

        self.embed_tokens = nn.Embedding(self.vocab_size,
                                         self.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(cfg)
                                     for _ in range(self.num_hidden_layers)])
        self.norm = RMSNorm(self.hidden_size, eps=cfg.norm_eps)
        self.out_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def prepare_attention_mask(input_shape, inputs_embeds,
                               past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if input_shape[-1] > 1:
            bsz, seq_len = input_shape
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min,
                              device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(dtype)

            if past_key_values_length > 0:
                mask = torch.cat([torch.zeros(
                    seq_len, past_key_values_length,
                    dtype=dtype, device=device), mask], dim=-1)

            return mask[None, None, :, :].expand(
                bsz, 1, seq_len, seq_len + past_key_values_length)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_length = input_ids.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        # input_embed: (batch_size, input_ids_len, hidden_size)
        inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self.prepare_attention_mask(
            (batch_size, seq_length), inputs_embeds, past_key_values_length)

        hidden_states = inputs_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] \
                if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache, )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.out_head(hidden_states)

        return logits

    def model_train(
            self,
            labels: Optional[torch.LongTensor] = None,):
        logits = self.forward()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss


if __name__ == '__main__':
    x = torch.randint(1, 500, (2, 50), dtype=torch.long)
    config = LlamaConfig()
    model = LlamaModel(config)
    y = model.forward(x).shape
    print(y)
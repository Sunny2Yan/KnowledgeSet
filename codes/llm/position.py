# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

from typing import Optional


# 1. learned position (bert, bart...)
class LearnedEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size)

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 在内存中是连续的并在序列化时导出
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")  # 绝对位置编码
        # position_ids = [[0, 1, ..., max_position_embeddings]]
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        token_type_ids: 用于标识当前token属于哪一个句向量（0属于第一句，1属于第二句）,
        用于辅助做NSP(next sentence prediction)任务
        past_key_values_length: 存储key,value并记录其长度
        """
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long,
                    device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings  # 直接加上可训练的position embedding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# 2. sine position (transformer, detr...)
class SinePositionEmbedding(nn.Module):
    """This is a more standard version of the position embedding."""

    def __init__(self, config, normalize=True, scale=None):
        super().__init__()
        self.embedding_dim = config.d_model // 2
        self.temperature = config.temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(
            self.embedding_dim,
            dtype=torch.float32,
            device=pixel_values.device)
        dim_t = self.temperature ** (
                2 * torch.div(dim_t, 2, rounding_mode="floor")
                / self.embedding_dim)

        # sin and cos position
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


# 3. ROPE (llama...)
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 计算位置编码的频率并加入缓存 $1 / 10000^(2k / d)$
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
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
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """将输入张量的一半隐藏维度进行旋转 (最后一个维度的左右调换).
    """
    x1 = x[..., : x.shape[-1] // 2]  # 最后一个维度的左半部分
    x2 = x[..., x.shape[-1] // 2:]  # 最后一个维度的右半部分
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # sin和cos的前两个维度总是1, 可以进行压缩.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# baichuan
def build_alibi_tensor(attention_mask: torch.Tensor,
                       num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
                        device=attention_mask.device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2,
                          device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    # 计算各个头的惩罚系数
    if closest_power_of_2 != num_heads:
        # 如果头数不是 2 的幂次方，修改惩罚系数
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device, dtype=torch.float32)
        num_remaining_heads = min(
            closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1, 1 + 2 * num_remaining_heads, 2,
            device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) *
                     attention_mask)[:, None, :]
    # 计算相对距离
    alibi = slopes[..., None] * arange_tensor
    # 计算 ALiBi 施加的注意力偏置
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


if __name__ == '__main__':
    x = torch.arange(24).expand((1, -1))
    print(x)
    y = x[:, 3 : 5 + 3]
    print(y)
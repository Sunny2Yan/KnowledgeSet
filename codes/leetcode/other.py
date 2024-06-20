# -*- coding: utf-8 -*-

class StringAlgorithm:
    def valid_ip_address(self, query_ip: str) -> str:
        """验证IP地址
        (leetcode 468) 给定一个字符串，如果符合IPv4格式，返回"IPv4"；如果符合IPv6格式，返回"IPv6"；否则返回"Neither"。
        思路：如果query中存在 . 按照ipv4判断; 如果query中存在 : 按ipv6判断; 否则都不是。
        ipv4：1) len=4 -> 2) x是数字；3) 0<=x<=255; 4) 长度大于1时，第一位不为0
        ipv6：1) len=8 -> 2) 1<=x.len()<=4; 3) x[i]是数字，或x[i] in [a-fA-F]
        时O(n); 空O(1)
        """
        if '.' in query_ip:
            ip_list = query_ip.split('.')
            if len(ip_list) != 4:
                return "Neither"
            for ip in ip_list:
                if not ip.isdigit() or int(ip) < 0 or int(ip) > 255 or (
                        len(ip) > 1 and int(ip[0]) == 0):
                    return "Neither"

            return "IPv4"

        elif ':' in query_ip:
            ip_list = query_ip.split(':')
            if len(ip_list) != 8:
                return "Neither"
            for ip in ip_list:
                if len(ip) < 1 or len(ip) > 4:
                    return "Neither"
                else:
                    s = ['a', 'b', 'c', 'd', 'e', 'f',
                         'A', 'B', 'C', 'D', 'E', 'F']
                    for i in ip:
                        if not i.isdigit() and i not in s:
                            return "Neither"
            return "IPv6"

        else:
            return "Neither"

# 最大无重复字串长度


import torch
import torch.nn as nn
from typing import Optional


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_tokens=2048, base=10000, device=None):
        super().__init__()
        # 计算位置编码的频率并加入缓存 $1 / 10000^(2k / d)$
        inv_freq = 1.0 / (base ** (torch.arange(
            0, dim, 2).float().to(device) / dim))
        t = torch.arange(
            self.max_tokens,
            device=inv_freq.device,
            dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # 向量外积
        emb = torch.cat((freqs, freqs), dim=-1)  # 矩阵横向拼接

        emb_cos = emb.cos()[None, None, :, :]  # 扩充维度
        emb_sin = emb.sin()[None, None, :, :]


        return emb_cos[:, :, :seq_len, ...].to(dtype=x.dtype), emb_sin[:, :, :seq_len, ...].to(dtype=x.dtype),


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_tokens):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_tokens = max_tokens

        assert self.head_dim * self.num_heads == self.hidden_size

        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size, bias=False)

        # self.rotary_emb = RotaryEmbedding(
        #     self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        # [1, 1, seq_len, head_dim]
        cos, sin = self.rotary_emb(self.head_dim, self.max_tokens)
        cos, sin = cos.to(dtype=value_states.dtype), sin.to(dtype=value_states.dtype)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, num_heads, seq_len, head_dim]

        # 如果有过去存储的键值对，则进行拼接
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states],
                                     dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)
                                    ) / math.sqrt(self.head_dim)
        assert attn_weights.size() == (
        bsz, self.num_heads, q_len, kv_seq_len)

        # 如果有注意力掩码，则进行掩码处理
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))

        # 计算注意力输出 (注意 fp32)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        assert attn_output.size() == (
        bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

    def rotary_emb(self, dim, max_token, base=10000, device=None, seq_len: int = None):
        inv_freq = 1.0 / (base ** (torch.arange(
            0, dim, 2).float().to(device) / dim))
        t = torch.arange(
            max_tokens,
            device=inv_freq.device,
            dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # 向量外积
        emb = torch.cat((freqs, freqs), dim=-1)  # 矩阵横向拼接

        emb_cos = emb.cos()[None, None, :, :]  # 扩充维度
        emb_sin = emb.sin()[None, None, :, :]

        return emb_cos[:, :, :seq_len, ...], emb_sin[:, :, :seq_len, ...]

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


if __name__ == '__main__':
    dim = 12
    base = 10000
    device = None
    max_tokens = 24
    inv_freq = 1.0 / (base ** (torch.arange(
        0, dim, 2).float().to(device) / dim))
    t = torch.arange(
        max_tokens,
        device=inv_freq.device,
        dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)  # 向量外积
    emb = torch.cat((freqs, freqs), dim=-1)  # 矩阵横向拼接

    x = emb.cos()[None, None, :, :]
    t = emb.sin()[None, None, :, :]
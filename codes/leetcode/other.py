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


import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


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

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        # [1, 1, seq_len, head_dim]
        cos, sin = self.rotary_emb(self.head_dim, self.max_tokens)
        cos = cos.to(dtype=value_states.dtype)
        sin = sin.to(dtype=value_states.dtype)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, num_heads, seq_len, head_dim]

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        assert attn_weights.size() == (bsz, self.num_heads, seq_len, kv_seq_len)

        # 如果有注意力掩码，则进行掩码处理
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, seq_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # 计算注意力输出 (注意 fp32)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        assert attn_output.size() == (bsz, self.num_heads, seq_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
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


class LinearRegression(object):
    def __init__(self, fit_intercept=True, if_standard=True,
                 epochs=10, eta=1e-2, batch_size=1, l1_ratio=None, l2_ratio=None):
        """
        :param fit_intercept: 是否训练bias
        :param solver:
        :param if_standard:  g
        """
        self.w = None
        self.fit_intercept = fit_intercept
        self.if_standard = if_standard
        if if_standard:
            self.feature_mean = None
            self.feature_std = None
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio

    def init_params(self, n_features):
        self.w = np.random.random(size=(n_features, 1))

    def _fit_sgd(self, x, y):
        """随机梯度下降求解"""
        x_y = np.c_[x, y]
        # 按 batch_size 更新 w,b
        for _ in range(self.epochs):
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // self.batch_size):
                batch_x_y = x_y[self.batch_size * index:self.batch_size * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]

                dw = -2 * batch_x.T.dot(batch_y - batch_x.dot(self.w)) / self.batch_size

                # 添加l1和l2的部分
                dw_reg = np.zeros(shape=(x.shape[1] - 1, 1))
                if self.l1_ratio is not None:
                    dw_reg += self.l1_ratio * self.sign_func(self.w[:-1]) / self.batch_size
                if self.l2_ratio is not None:
                    dw_reg += 2 * self.l2_ratio * self.w[:-1] / self.batch_size
                dw_reg = np.concatenate([dw_reg, np.asarray([[0]])], axis=0)
                dw += dw_reg
                self.w = self.w - self.eta * dw

    def fit(self, x, y):
        # 是否归一化feature
        if self.if_standard:
            self.feature_mean = np.mean(x, axis=0)
            self.feature_std = np.std(x, axis=0) + 1e-8
            x = (x - self.feature_mean) / self.feature_std
        # 是否训练bias
        if self.fit_intercept:
            x = np.c_[x, np.ones_like(y)]
        # 初始化参数
        self.init_params(x.shape[1])
        # 训练模型
        self._fit_sgd(x, y)


# RNN实现
class CustomRNN:
    def __init__(self, batch_size, seq_len, input_size, hidden_size, is_bidirect):
        self.batch_size, self.seq_len = 2, 3  # batch size, sequence length
        self.input_size, self.hidden_size = 2, 3
        self.is_bidirect = is_bidirect

        # x = torch.randn(bs, L, input_size)
        self.h_prev = torch.zeros(batch_size, hidden_size)
        self.bi_h_prev = torch.zeros(2, batch_size, hidden_size)
        self.rnn, self.bi_rnn = None, None

    def api_mode(self, x: torch.Tensor):
        """The api model in torch
        :param x: a tensor of shape [batch_size, seq_len, input_size]
        :return: bi_rnn_out, bi_state_final
        """
        self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        self.bi_rnn = nn.RNN(self.input_size, self.hidden_size,
                             batch_first=True, bidirectional=True)
        rnn_out, state_final = self.rnn(x, self.h_prev.unsqueeze(0))
        bi_rnn_out, bi_state_final = self.bi_rnn(x, self.bi_h_prev)

        if not self.is_bidirect:
            return rnn_out, state_final
        else:
            return bi_rnn_out, bi_state_final

    def function_mode(self, x: torch.Tensor):
        """
        :param x: a tensor of shape [batch_size, seq_len, input_size]
        :return: bi_rnn_out, bi_state_final
        """
        custom_rnn_out, custom_state_final = self.rnn_forward(
            x, self.rnn.weight_ih_l0, self.rnn.weight_hh_l0,
            self.rnn.bias_ih_l0, self.rnn.bias_hh_l0, self.h_prev)

        custom_bi_rnn_out, custom_bi_state_final = self.bidirectional_rnn_forward(
            x, self.bi_rnn.weight_ih_l0, self.bi_rnn.weight_hh_l0,
            self.bi_rnn.bias_ih_l0, self.bi_rnn.bias_hh_l0, self.bi_h_prev[0],
            self.bi_rnn.weight_ih_l0_reverse, self.bi_rnn.weight_hh_l0_reverse,
            self.bi_rnn.bias_ih_l0_reverse, self.bi_rnn.bias_hh_l0_reverse,
            self.bi_h_prev[1])

        if not self.is_bidirect:
            return custom_rnn_out, custom_state_final
        else:
            return custom_bi_rnn_out, custom_bi_state_final

    @staticmethod
    def rnn_forward(x, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
        batch_size, seq_len, input_size = x.shape
        h_dim = weight_ih.shape[0]
        h_out = torch.zeros(batch_size, seq_len, h_dim)  # 状态矩阵

        for sl in range(seq_len):
            x_ = x[:, sl, :].unsqueeze(2)  # 当前时刻特征 [bs, input_size, 1]
            # [bs, h_dim, input_size]
            w_ih_batch = weight_ih.unsqueeze(0).tile(batch_size, 1, 1)
            w_hh_batch = weight_hh.unsqueeze(0).tile(batch_size, 1, 1)

            # [bs, h_dim]
            w_times_x = torch.bmm(w_ih_batch, x_).squeeze(-1)
            w_times_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)
            h_prev = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh)

            h_out[:, sl, :] = h_prev

        return h_out, h_prev.unsqueeze(0)

    def bidirectional_rnn_forward(
            self, x, weight_ih, weight_hh, bias_ih, bias_hh, h_prev,
            weight_ih_reverse, weight_hh_reverse, bias_ih_reverse,
            bias_hh_reverse, h_prev_reverse):
        batch_size, seq_len, input_size = x.shape
        h_dim = weight_ih.shape[0]
        h_out = torch.zeros(batch_size, seq_len, h_dim * 2)  # 状态矩阵

        forward_out = self.rnn_forward(
            x, weight_ih, weight_hh, bias_ih, bias_hh, h_prev)[0]
        backward_out = self.rnn_forward(
            torch.flip(x, [1]), weight_ih_reverse, weight_hh_reverse,
            bias_ih_reverse, bias_hh_reverse, h_prev_reverse)[0]

        h_out[:, :, :h_dim] = forward_out
        h_out[:, :, h_dim:] = backward_out

        return h_out, h_out[:, -1, :].reshape((batch_size, 2, h_dim)).transpose(0, 1)


if __name__ == '__main__':
    dim = 12
    base = 10000
    device = None
    max_tokens = 24
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(
        max_tokens,
        device=inv_freq.device,
        dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)  # 向量外积
    emb = torch.cat((freqs, freqs), dim=-1)  # 矩阵横向拼接

    x = emb.cos()[None, None, :, :]
    t = emb.sin()[None, None, :, :]
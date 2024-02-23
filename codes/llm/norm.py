# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# bert, bart, GPT2
# LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # [bsz, max_len, hidden_dim]
        mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
        return self.w * (x - mean) / (std + self.eps) + self.b


# llama, baichuan
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

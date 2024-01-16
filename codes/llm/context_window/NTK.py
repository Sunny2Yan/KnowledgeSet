import torch
import transformers
import numpy as np
from transformers.models.llama import modeling_llama


# class NTKLlamaRotaryEmbedding(torch.nn.Module):
#     def __init__(self, dim, max_position_embeddings=16384, base=10000, alpha=8, device=None):
#         super().__init__()
#         alpha = alpha
#         base = base * alpha ** (dim / (dim-2))
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
#         self.register_buffer("inv_freq", inv_freq)
#
#         # Build here to make `torch.jit.trace` work.
#         self.max_seq_len_cached = max_position_embeddings
#         t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
#         self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
#
#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
#         if seq_len > self.max_seq_len_cached:
#             self.max_seq_len_cached = seq_len
#             t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             # Different from paper, but it uses a different permutation in order to obtain the same calculation
#             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
#             self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
#             self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#         )
#
#
# transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = NTKLlamaRotaryEmbedding
#
#
#
# # 苏剑林版本
# def ntk_rope_mixed_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
#     old_init(self, dim, max_position_embeddings, base, device)
#     k, b = 12, 0.75
#     max_position_embeddings = training_length * k
#     a = np.log(k) / (dim / 2)**b
#     inv_freq = base**(-torch.arange(0, dim, 2).float().to(device) / dim)
#     inv_freq *= (-a * torch.arange(1, dim // 2 + 1).float().to(device)**b).exp()
#     self.register_buffer('inv_freq', inv_freq)
#     self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())
#
#
# def apply_rotary_pos_emb_and_logn_scale(q, k, cos, sin, position_ids):
#     q_embed, k_embed = old_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
#     scale = ((position_ids + 1)[:, None, :, None].log() / np.log(training_length)).clip(1)
#     return q_embed * scale.to(q_embed.dtype), k_embed
#
#
# training_length = 4096
# old_init = modeling_llama.LlamaRotaryEmbedding.__init__
# old_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
# modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_rope_mixed_init
# modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb_and_logn_scale


# 官方版
old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):

    # The method is just these three lines
    max_position_embeddings = 16384
    a = 8  # Alpha value
    base = base * a ** (dim / (dim-2))  # Base change formula

    old_init(self, dim, max_position_embeddings, base, device)
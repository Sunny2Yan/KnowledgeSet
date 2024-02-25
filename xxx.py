import torch

# base = 5
# dim = 10
# device = torch.device('cuda')
#
# inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
#
# max_seq_len_cached = 10
#
# # 构建位置编码矩阵
# t = torch.arange(max_seq_len_cached,
#                  device=inv_freq.device,
#                  dtype=inv_freq.dtype)
# freqs = torch.einsum("i,j->ij", t, inv_freq)  # 向量外积
# # print(freqs.shape)
#
# emb = torch.cat((freqs, freqs), dim=-1)
#
# # print(emb)
# # print(emb.cos()[None, None, :, :].shape)
#
# emb = torch.rand(2, 2, 10, 10)
# print(emb)
# x1 = emb[..., : emb.shape[-1] // 2]
# x2 = emb[..., emb.shape[-1] // 2:]
# print(torch.cat((-x2, x1), dim=-1))
x = torch.rand(2, 3).to(None)
y = torch.max(x, torch.tensor(torch.finfo(x.dtype).min))
print(x)
print(y)
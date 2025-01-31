# DeepSeek 模型结构

## 1. deepseek V3

### 1.1 多头潜在注意力机制（Multi-Head Latent Attention）

![](/imgs/llm/deepseek/mla.png)

对于 input hidden $h_t$，MLA计算如下：
$$
\begin{array}{r|l}
\begin{aligned}
    Latent ~ c_t^{KV} &= W^{DKV} h_t \\
    [k_{t,1}^C, k_{t,2}^C, \cdots, k_{t,n_h}^C] = k_t^C &= W^{UK} c_t^{KV} \\
    k_t^R &= \mathrm{RoPE}(W^{KR} h_t) \\
    k_{t,i} &= [k_{t,i}^C, k_t^R] \\
    [v_{t,1}^C, v_{t,2}^C, \cdots, v_{t,n_h}^C] = v_t^C &= W^{UV} c_t^{KV}
\end{aligned} 
&
\begin{aligned}
    Latent ~ c_t^{Q} &= W^{DQ} h_t \\
    [q_{t,1}^C, q_{t,2}^C, \cdots, q_{t,n_h}^C] = q_t^C &= W^{UQ} c_t^{Q} \\
    [q_{t,1}^R, q_{t,2}^R, \cdots, q_{t,n_h}^R] = q_t^R &= \mathrm{RoPE}(W^{QR} c_t^Q) \\
    q_{t,i} &= [q_{t,i}^C, q_{t,i}^R] \\
\end{aligned}
\end{array}
$$

其中，$h_t\in \mathbb{R}^d$ 表示输入的第 $t$ 个token，$n_h$ 表示 attention head 数, $d_h$ 表示每个head attention的维度。
- $W^{DKV} \in \mathbb{R}^{d_c \times d}, W^{UV} \in \mathbb{R}^{n_hd_h \times d_c}$ 分别表示下采样和上采样矩阵($d_c \ll n_hd_h$)
- $W^{KR} \in \mathbb{R}^{d_h^R \times d}$ 用于生成携带旋转位置编码的解耦键
- $W^{DQ} \in \mathbb{R}^{d'_c}, W^{UQ} \in \mathbb{R}^{n_hd_h \times 𝑑'_c}$ 分别表示下采样和上采样矩阵（$d'_c \ll n_hd_h$）
- $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d'_c}$ 用于生成携带旋转位置编码的解耦查询矩阵

值得注意的是，在 KV cache缓存时只需要缓存 $c_t^{KV}$ 和 $k_t^R$，大大降低缓存空间。


$$
\begin{aligned}
o_{t,i} &= \sum_{j=1}^{t} \mathrm{Softmax}_j \frac{(q_{t,i}^T k_{j,i})}{\sqrt{d_h + d_h^R}} v_{j,i}^C \\
u_t &= W^{O} [o_{t,1}, o_{t,2}, \cdots, o_{t,n_h}]
\end{aligned}
$$

```python
import math
import torch
import torch.nn as nn

class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads  # 16
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # 128 + 64
        self.v_head_dim = args.v_head_dim

        self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
            
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq(x)  # compression q: [bs, seq, 2048] -> [bs, seq, 16*512]
            
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)  # [bs, seq, 16, 128+64]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)  # compression kv: [bs, seq, 2048] -> [bs, seq, 512+64]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        
        x = self.wo(x.flatten(2))
        return x
```


### 1.2 deepseek MoE 架构
令 $u_t$ 表示 FFN 输入的第 $t$ 个token，计算 FFN 的输出 $h_t^{'}$ 如下：

$$
\begin{aligned}
    h_t^{'} &= u_t + \sum_{i=1}^{N_s} \mathrm{FFN}_i^{(s)}(u_t) + \sum_{i=1}^{N_r} g_{i,t} \mathrm{FFN}_i^{(r)}(u_t) \\
    g_{i,t} &= \frac{g_{i,t}^{'}}{\sum_{j=1}^{N_r} g_{j,t}^{'}} \\
    g_{j,t}^{'} &=  
    \begin{cases}
    s_{i,t}  & s_{i,t} \in \mathrm{Topk}(\{s_{i,t}|1\leq j \leq N_r\}, K_r) \\
    0 & otherwise
    \end{cases} \\
    s_{i,t} &= \mathrm{Sigmoid}(u_t^T e_i)
\end{aligned}
$$

- $N_s, N_r$ 分别表示 shared experts 和 routed experts 的数量
- $\mathrm{FFN}_i^{(s)}(\cdot), \mathrm{FFN}_i^{(r)}(\cdot)$ 分别表示第 $i$ 个 shared experts 和 routed experts
- $K_r$ 表示 active routed experts 的数量
- $g_{i,t}$ 表示第 $i$ 个专家的权重系数
- $s_{i,t}$ 表示 token 与专家的相关度
- $e_i$ 表示第 $i$ 个专家的特征向量
- $\mathrm{Topk}(\cdot, K)$ 表示第 $t$ 个 token 与所有 routed exports 计算得到的相关度得分最高的 $K$ 个值

### 1.3 无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing）


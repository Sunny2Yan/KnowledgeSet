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
- $s_{i,t}$ 表示 token 与专家的亲和度
- $e_i$ 表示第 $i$ 个专家的特征向量
- $\mathrm{Topk}(\cdot, K)$ 表示第 $t$ 个 token 与所有 routed exports 计算得到的相关度得分最高的 $K$ 个值

### 1.3 无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing）

解决 MoE 负载均衡问题，传统的解决方法是依赖于辅助损失，但过大的损失会降低模型性能，本方法为每个专家引入了一个偏置项，作为相应的亲和度得分，以确定 top-K 路由：
$$
g_{i,t}^{'} = 
\begin{cases}
s_{i,t}  & s_{i,t}+b_i \in \mathrm{Topk}(\{s_{i,t}+b_i|1\leq j \leq N_r\}, K_r) \\
0 & otherwise
\end{cases}
$$
偏置项仅用于路由选择，gating value 仍基于原始亲和度得分。
即，训练过程中，实时监控每个训练步骤中所有批次的专家负载分布，步骤结束时，负载过高的专家偏置项会减少 $\gamma$，负载不足的专家偏置项增加 $\gamma$。

### 1.4 序列级辅助损失补充机制

### 1.5 Node-Limited Routing
目的：在训练期间限制通信成本。即，根据分布在每个节点上的专家的最高亲和度分数之和 $\frac{K_r}{M}$ 来选择，确保每个token将被发送到最多 M 个 GPU node。

## 2. 多token预测（Multi-Token Prediction，NTP）
![](/imgs/llm/deepseek/mtp.png)

目的：降低 training cast -> 方法：speculative decoding -> 提高它的性能。
方法：使用 D sequential modules 来预测 D 个额外的 tokens。MTP 模块包含与主模型共享的 embedding 层 $\mathrm{Emb}(\cdot)$、输出头 $\mathrm{OutHead}(\cdot)$ 和 
Transformer 块 $\mathrm{TRM}_k(\cdot)$、投影矩阵 $M_k \in \mathbb{R}^{d \times 2d}$。
 
记第 $i$ 个 token 在第 $k-1$ 个预测深度为 $h_i^{k-1} \in \mathbb{R}^d$（$k=1$ 表示 main model），则对于第 $i$ 个输入token $t_i$，在第 $k$ 个预测深度：
$$
\begin{aligned}
h_i^{'k} &= M_k [\mathrm{RMSNorm}(h_i^{k-1}); \mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))] \\
h_{i:T-k}^k &= \mathrm{TRM}_K(h_{i:T-k}^{'k}) \\
p_{i+k+1}^k &= \mathrm{Softmax}(\mathrm{OutHead}(h_i^k))
\end{aligned}
$$

其中，$T$ 表示 sequence length。

**对于训练：**

$$
\begin{aligned}
\mathcal{L}_{MTP}^{k} &= \mathrm{CrossEntropy}(p_{2+k:T+1}^k, t_{2+k:T+1}) = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log p_i^t [t_i] \\
\mathcal{L}_{MTP} &= \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{MTP}^{k}
\end{aligned}
$$

**对于推理：**

MTP 策略主要是为了提高 main model 的性能，因此在推理过程中，可以直接丢弃 MTP 模块，独立运行 main model。


```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class share_embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size
            )
    def forward(self,input_ids):
        return self.embed_tokens(input_ids)
        
class share_output_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(self,transformer_hidden):
        return self.lm_head(transformer_hidden)
        
class transformer_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mla = MLA(config)
        self.moe = MOE(config)
        self.pre_RMSNorm = RMSNorm(config)
        self.post_RMSNorm = RMSNorm(config)
        
    def forward(self, input_hidden):
        out_logits = self.pre_RMSNorm(input_hidden)
        out_logits = self.mla(out_logits)
        out_logits = self.post_RMSNorm(out_logits)        
        moe_input = input_hidden+out_logits
        moe_output = self.moe(moe_input)
        block_output = moe_output+moe_input
        return block_output
        
class MTP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.RMSNorm_right = RMSNorm(config)
        self.RMSNorm_left = RMSNorm(config)
        self.transformer = transformer_block(config)
        self.proj = nn.Linear(2*config.hidden_size,config.hidden_size)
    def forward(self,last_block_out,input_ids_truncated,share_embeding,share_lm_head):
        last_norm_out = self.RMSNorm_right(last_block_out)
        embeding_trunc = share_embeding(input_ids_truncated)
        concat_input = torch.cat((last_norm_out, embeding_trunc), dim=-1)
        proj_out = self.proj(concat_input)
        trans_out_logits = self.transformer(proj_out)
        return trans_out_logits
        
class MTP_and_deepseek(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.share_embedding = share_embedding(config)
        self.share_lm_head = share_output_head(config)
        self.loss = nn.CrossEntropyLoss(ignore = config.pad_token_id)
        self.Model_trans_blocks = nn.ModuleList(
            [
                transformer_block(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.MTP_trans_blocks = nn.ModuleList(
            [
                transformer_block(config, layer_idx)
                for layer_idx in range(config.num_MTP_layers)
            ]
        )
        self.config = config
        self.alpha_list = config.alpha_list  # 对于每一个MTP loss的加权列表
        
   def forward(self, input_ids):
       # input_ids: [bos,tok1,tok2......last_tok]
       # labels_origin: [tok1,tok2,....,last_tok,eos/pad]
       # labels_MTP1: [tok2,.....last_tok,eos,pad]
       # labels_MTP2: [tok3,.....last_tok,eos,pad,pad]
       embeding_logits = self.share_embedding(input_ids)
       deepseek_hidden = embeding_logits
       for index,trans_block in enumerate(self.Model_trans_blocks):
           deepseek_hidden = trans_block(deepseek_hidden)
       deepseek_logits = self.share_lm_head(deepseek_hidden)
       labels = torch.cat([input_ids[:, 1:], 
                           torch.full((input_ids.size(0), 1), self.config.pad_token_id)], dim=1)
       Main_loss = self.loss(deepseek_logits,labels)
       
       last_mtp_out = deepseek_hidden
       for ind, MTP in enumerate(self.MTP_trans_blocks):
           input_ids_trunc = torch.cat([input_ids[:, ind + 1 :], torch.full((input_ids.size(0), ind + 1), self.config.pad_token_id)], dim=1, )
           mtp_out = MTP(last_mtp_out, input_ids_trunc,self.share_embedding)
           mtp_logits = self.share_lm_head(mtp_out)
           last_mtp_out = mtp_out
           labels_trunc = torch.cat([input_ids_trunc[:, 1:], torch.full((input_ids.size(0), 1), config.pad_token_id)], dim=1)
           mtp_loss = self.loss(mtp_logits,labels_trunc)
           alpha = self.alpha_list[ind]
           Main_loss +=  alpha*mtp_loss
       return Main_loss
```

# 推理优化

## 批处理（Batching）
提高 GPU 利用率和有效吞吐量的最简单方法是通过批处理，但批量太大时可能会导致内存溢出。

## K-V Cache
为了避免在每个时间步重新计算所有tokens的这些张量，可以将它们缓存在 GPU 内存中.

在推理时，由于是预测下一个token，则下一个step的输入就包含了上一个step的内容，只是末尾多了一个token。
那么下一个step的计算也应该包含上一个step的计算，于是 KV_Cache=[(k_0,v_0), (k_1,v_1), ...]。

$$
i 时刻：
q_i = x_iW_q; k_i = x_iW_k; v_i=x_iW_v  \\
a_{ij} = \frac{\exp(q_ik_j^T / \sqrt{d})}{\sum_{t=1}^i \exp(q_ik_t^T / \sqrt{d})} \\
o_i = \sum_{j=1}^i a_{ij} v_j  \\

i+1 时刻：
q_{i+1} = x_{i+1}W_q; k_{i+1} = x_{i+1}W_k; v_{i+1}=x_{i+1}W_v  \\
a_{i+1, j} = \frac{\exp(q_{i+1}k_j^T / \sqrt{d})}{\sum_{t=1}^{i+1} \exp(q_{i+1}k_t^T / \sqrt{d})} \\
o_{i+1} = \sum_{j=1}^{i+1} a_{i+1, j} v_j
$$
可以发现 $k_1, \cdots, k_i$ 和 $v_1, \cdots, v_i$ 在之前都算过了，没有必要重新计算。可以用空间换时间，把它们缓存起来.

对于输入长度为 S，层数为 L，hidden_size为 d 的模型，需要缓存的参数量为：$2*s*d*L$
以llama 7B为例：L=32, hidden_size=4096, s=1k时：2*1024*4096*32=268435456 (512M)

## Attention 机制
1. 多头注意力(MHA, Multi Head Attention) 
   思想：利用 KV Cache 以空间换时间
   操作：输入分别经过 W_q、W_k、W_v 的变换之后，都切成了 num_head 份，维度也从 d_model 降到了d_head，再分别对每个head进行attention计算并拼接
   ```python
   qkw_1 = nn.Linear(d_model, 3*d_model)
   q_1, k_1, v_1 = qkv(x).chunk(3, dim=2)
   ```
2. 多查询注意力(MQA, Multi Query Attention)
   思想：多个注意力头之间共享 K, V，来降低 K-V Cache，以减少空间消耗，但性能会有所降低
   操作：经过 W_q、W_k、W_v 的变换后只对 Q 进行切分，而 K、V直接在线性变换的时候把维度降到了d_head，然后这 n 个 Query 头分别和同一份 K、V 进行 attention 计算，之后把结果拼接起来。
   ```python
   qkw_2 = nn.Linear(d_model, d_model + 2*d_head)  # d_model = num_head * d_head
   q_2, k_2, v_2 = qkv(x).split([d_model, d_head, d_head], dim=2)
   ```
3. 分组注意力(GQA, Group Query Attention) 
   思想：对上述 MHA 与 MQA 取折中方案。如：llama2,3
   操作：经过 W_q、W_k、W_v 的变换后 Q 仍不变，而 K、V 在线性变换的时候把维度降到了 group*d_head，同一个 group 内的 Q 共享同一套 K、V，不同 group 的 Q 所用的 K、V 不同
   ```python
   group_head = num_head / group
   qkw_3 = nn.Linear(d_model, d_model + 2*group_head*d_head)
   q_3, k_3, v_3 = qkv(x).split([d_model, group_head*d_head, group_head*d_head], dim=2)
   ```

## Flash Attention
[papaer](https://arxiv.org/pdf/2205.14135); [参考](https://fancyerii.github.io/2023/10/23/flashattention/)

(内存, 带宽)：CPU DRAM(1T, 12.8GB/S) -> GPU HBM(40GB, 1.5TB/S) -> GPU SRAM(20MB, 19TB/S)
标准的 Attention：
1. 从 HBM 中载入 Q, K，计算注意力分数 S，并将 S 写入 HBM 中；
2. 从 HBM 中载入注意力分数 S，计算概率 P=softmax(S)，并将 P 写入 HBM 中；
3. 从 HBM 中载入 P, V，计算输出 O=PV，并将 O 写入 HBM 中。

缺点：每次都需要从 HBM 把数据加载的 GPU 的 SRAM 中做运算，运算结束后又从 SRAM 复制到 HBM 中。
思想：避免数据的来回移动，直接在 SRAM 中做运算，但需要降低数据的存储空间。
操作：1) 将 NxN 的 softmax(S) 矩阵划分为块; 2) activation/gradient checkpointing重计算。

$softmax(x)=\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i-m(x)}}{\sum_j e^{x_j-m(x)}}$，其中 $m(x)=\max(x)$ 是为了计算的数值稳定。于是
$$
f(x)=[e^{x_i-m(x)}, \cdots, e^{x_n-m(x)}]\\ 
l(x)=\sum_j f(x) \\ 
softmax(x)=\frac{f(x)}{l(x)}
$$
对 $x$ 分块 $x=[x^{(1)}, x^{(2)}]$，则有: 
$$
m(x)=\max(m(x^{(1)}), m(x^{(2)})) \\
f(x)=[e^{m(x^{(1)})-m(x)}f(x^{(1)}), e^{m(x^{(2)})-m(x)}f(x^{(2)})] \\
l(x)=e^{m(x^{(1)})-m(x)}l(x^{(1)}) + e^{m(x^{(2)})-m(x)}l(x^{(2)})]
$$

设 SRAM大小为 M，具体操作如下：
设置块的行 $B_r=\frac{M}{4d}$ ，块的列 $B_c=\min(\frac{M}{4d}, d)$。min函数防止块大小 $B_r×B_c>\frac{M}{4}$；
在 HBM 中初始化 $O=(0)_{N\times d}, l=(0)_N, m=(-\infty)_{N}$，之后会逐步把中间结果累加进去；
将 Q 划分成 $B_r \times d$ 大小的 $T_r=\frac{N}{B_r}$ 块，将 K, V 划分为 $B_c \times d$ 大小的 $T_c=\frac{N}{B_c}$ 块 -> $QK^TV \in R^{B_r \times d}$；
将 O 划分成大小为 $B_r\times d$ 的 $T_r$ 块，将 l 向量划分为大小为 $B_r$ 的 $T_r$ 块，将 m 向量划分为大小为 $B_r$ 的 $T_r$ 块;
for $0 <= j <= T_c$:
   $\;\;\;\;$ 从 HBM 中载入 $K_j, V_j$ 到 SRAM 中；
   $\;\;\;\;$ for $0 <= i <= T_r$:
   $\;\;\;\;\;\;\;\;$ 从 HBM 中载入 $Q_i, O_i, l_i, m_i$ 到 SRAM 中；
   $\;\;\;\;\;\;\;\;$ 计算注意力分数 $S_{ij}=Q_iK_{j}^T \in R^{B_r\times B_c}$
   $\;\;\;\;\;\;\;\;$ 计算 $\tilde{m}_{ij}=rowmax(S_{ij}) \in R^{B_r}; \tilde{P}_{ij}=\exp(S_{ij}-\tilde{m}_{ij})\in R^{B_r\times B_c}; \tilde{l}_{ij}=rowmax(\tilde{P}_{ij}) \in R^{B_r}$
   $\;\;\;\;\;\;\;\;$ 计算 $m_i^{new}=\max(m_i, \tilde{m}_{ij}) \in R^{B_r}; l_i^{new}=e^{m_1-m_i^{new}}l_i + e^{\tilde{m}_{ij}-m_i^{new}}\tilde{l}_{ij} \in R^{B_r}$;
   $\;\;\;\;\;\;\;\;$ 更新 $O_i \leftarrow diag(l_i^{new})^{−1}(diag(l_i)e^{m_i−m_i^{new}}O_i + e^{\tilde{m}_{ij}-m_i^{new}}\tilde{P}_{ij}V_j)$，并写入 HBM中；
   $\;\;\;\;\;\;\;\;$ 更新 $l_i \leftarrow l_i^{new}; m_i \leftarrow m_i^{new}$，并写入 HBM；
输出：O

## Paged Attention
[paper](https://arxiv.org/pdf/2309.06180); [参考](https://fancyerii.github.io/2023/11/01/pagedattention/)

目的： 管理显存，避免显存碎片和重复造成浪费，提高吞吐量。基于此实现了vLLM。

把每个 Sequence (每个请求的 prompt+response)的 KV cache 切分成一系列 KV block。 
block 包含固定大小 token 的 KV Cache。设 block 的大小是 B。定义第 j 个块中的 B 个 key 为 $K_j=(k_{(j−1)B+1}, \cdots, k_{jB})$，
B 个 value 为 $V_j=(v_{(j−1)B+1}, \cdots, v_{jB})$ 。那么就可以一次计算 token i 对整个第 j 块的 B 个 token 的 attention：
$$
A_{ij} = \frac{\exp(q_iK_j^T / \sqrt{d})}{\sum_{t=1}^{i/B} \exp(q_iK_t^T 1_{t\in B} / \sqrt{d})};   \\
o_i=\sum_{j=1}^{i/B} A_{ij}V_j
$$

## 模型量化（Quantization）

## 矩阵稀疏（Sparsity）
稀疏矩阵是许多元素为 0 的矩阵，即对模型修剪或用 0 替换某些接近 0 的值。然后这些矩阵可以用压缩形式表示。

## 蒸馏（Distillation）
将大模型知识转移到较小的模型上。
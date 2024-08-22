# Mamba

## 1. SSM (State Space Model)
状态空间模型 (Structured state space sequence models, S4) 被定义为 $(\Delta, A, B, C)$;

$$
h(x) = Ah(t) + Bx(t) \Longrightarrow h_t=\bar{A}h_{t-1} + Bx(t) \Longrightarrow \bar{K}=(C\bar{B}, C\bar{AB}, \dots, C\bar{X}^k\bar{b}, \dots) \\
y(t) = Ch(t) \Longrightarrow y=Ch_t \Longrightarrow y=x * \bar{K}
$$

线性时间不变性 (Linear Time Invariance, LTI):

$A\in \R^{N \times N}, B\in \R^{N\times 1}, C\in \R^{1\times N}$
对于输入序列 $x$, batch_size=B, sequence_len=L, channel($\Leftrightarrow head$)=D, 则 hidden_size=DN, 因此时间空间复杂度都为 $O(BLDN)$

1. 离散化 (Discretization): $(\Delta, A, B, C) \rightarrow (\bar{A}, \bar{B}, C)$
   discretization rule: $\bar{A} = f_{A}(\Delta, A)$; $\bar{B} = f_{B}(\Delta, B)$
   the zero-order hold (ZOH): $\bar{A} = \exp(\Delta A)$; $\bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B$
2. 计算 (Computation): 
   线性递归 (linear recurrence): 输入每次看到一个时间步长；
   全局卷积 (global convolution): 整个输入序列可以提前看到，因此可以并行训练。

## 2. Selective State Space Models (S6)

Motivation:
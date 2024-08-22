# Mamba
RNN：$h_t = f(Ux_i + Wh_{i-1}); y_i = softmax(Vh_i)$ 
特点：1）串行导致训练速度慢（无法并行），推理速度快；2）遗忘问题，无法处理长度依赖关系。

CNN（一维）：
特点：1）可以并行计算；2）关注局部特征而忽视全局特征；3）固定卷积核限制推理速度。

Attention：
特点：1）可以并行，能够捕捉长度依赖关系；2）时间空间复杂度为 O(n^2)；3）推理速度慢。



## 1. SSM (State Space Model)
状态空间模型 (Structured state space sequence models, S4) 被定义为 $(\Delta, A, B, C)$;

$$
h(x) = Ah(t) + Bx(t) \Longrightarrow h_t=\bar{A}h_{t-1} + Bx(t) \Longrightarrow \bar{K}=(C\bar{B}, C\bar{AB}, \dots, C\bar{X}^k\bar{B}, \dots) \\
y(t) = Ch(t) \Longrightarrow y=Ch_t \Longrightarrow y=x * \bar{K}
$$
此时，若层层递归计算，则同RNN；若将 $\bar{B}$ 视为卷积核，则可进行并行运算。

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
由于 RNN 与 CNN 的长期依赖捕捉能力都不行，因此：

使用HIPPO矩阵代替随机初始化的 A 矩阵，可以有效缓解遗忘问题。即：

A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2}  # below the diagonal
       = n+1  # the diagonal
       = 0  # above the diagonal

为了避免 HIPPO 矩阵的空间复杂度，可以使用低秩矩阵表示：

$$


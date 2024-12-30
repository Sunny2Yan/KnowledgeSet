# Flash Attention

![](/imgs/llm/flash_att/f_a_1.png)

## 1. Standard Attention 
#### Forward
给定输入序列 $Q, K, V \in \mathbb{R}^{N \times d}$，其中 $𝑁$是序列长度、$𝑑$ 是 head dimension，通过下面公式计算 attention 输出 $O \in \mathbb{R}^{N \times d}$：
$$
\begin{align}
S &= QK^{T} \in \mathbb{R}^{N \times N} \\
P &= softmax(S) \mathbb{R}^{N \times N} \\
O &= PV \in \mathbb{R}^{N \times d} \\
\end{align}
$$

![](/imgs/llm/flash_att/f_a_2.png)

缺点：由于 SRAM 空间较小，这样直接大量的读写导致 Attention 运算速度较慢，而且会有内存碎片化问题。

#### Backward
给定输入序列 $Q, K, V \in \mathbb{R}^{N \times d}$，输出 $O \in \mathbb{R}^{N \times d}$，输出的梯度 $dO$，来计算输入的梯度 $dQ, dK, dV \in \mathbb{r}^{N \times B}$：

![](/imgs/llm/flash_att/f_a_3.png)

## 2. Flash Attention V1

#### Kernel fusion
从 HBM 加载输入，执行所有计算步骤（矩阵乘法、softmax、可选的掩码和 dropout、矩阵乘法），然后将结果写回 HBM。而不是从HBM反复读写。

#### Recomputation
向前的时候 P 和 S 都不会存起来，在向后的时候，再计算一次向前把P和S再算出来然后执行向后，求 QKV 的梯度，即共执行了2次前进1次后向。

#### Tiling
对于一个向量 $x \in \mathbb{R}^B$，softmax 计算如下：
$$
m(x) := \max_{i} x_i, ~~~~
f(x) := [e^{x_1-m(x)}, \cdots, e^{x_B-m(x)}],~~~~
l(x) := \sum_{i}f(x)_i, ~~~~
softmax(x) := \frac{f(x)}{l(x)}
$$

对于向量 $x^(1), x^(2) \in \mathbb{R}^B$，$x=[x^{(1)}, x^{(2)}] \in \mathbb{R}^{2B}$，计算 softmax 如下：
$$
\begin{align}
m(x) &= m(x^{(1)}, x^{(2)}) = \max(m(x^{(1)}), m(x^{(2)})), ~~~~
f(x) = [e^{m(x^{(1)}) - m(x)}f(x^{(1)}), e^{m(x^{(2)}) - m(x)}f(x^{(2)})], \\
l(x) &= l([x^{(1)}, x^{(2)}]) = e^{m(x^{(1)}) - m(x)}l(x^{(1)}) + e^{m(x^{(2)}) - m(x)}l(x^{(2)}), ~~~~
softmax(x) = \frac{f(x)}{l(x)}
\end{align}
$$

注意：减 $m(x)$，是为了数值稳定，结果保持不变

### 前向算法：
![](/imgs/llm/flash_att/f_a_4.png)
注意：$B_r$ 中的 $\min()$ 是为了防止 $B_r \times B_c > M/4$。矩阵分割只按行分割，列数保持不变。

### 后向算法：

![](/imgs/llm/flash_att/f_a_5.png)


## 3. Flash Attention V2
FlashAttention 仍不如优化的矩阵乘法 (GEMM) 操作快，仅达到理论最大 FLOP/s 的 25-40%。导致效率低下是由于不同线程块和 GPU 上的扭曲之间的次优工作分区，导致低占用或不必要的共享内存读取/写入。

1. 调整算法以减少非矩阵乘法FLOPs数量

在 Flash Attention 中：
$$
O^{(2)} = diag(l^{(1)} / l^{(2)})^{-1} O^{(1)} + diag(l^{(2)})^{-1} e^{S^{(2)} - m^{(2)}} V^{(2)}
$$

$$
\tilde{O}^{(2)} = diag(l^{(1)})^{-1} O^{(1)} + e^{S^{(2)} - m^{(2)}}V^{2}
$$

2. 并行计算注意力，即使是单个头部，也跨不同的线程块以增加占用率
3. 每个线程块内，将工作分配给线程束以减少通过共享内存的通信


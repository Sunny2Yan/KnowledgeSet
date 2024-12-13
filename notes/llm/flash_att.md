# Flash Attention

![](/imgs/llm/flash_att/f_a_1.png)

## 1. Standard Attention 
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

## 2. Flash Attention V1

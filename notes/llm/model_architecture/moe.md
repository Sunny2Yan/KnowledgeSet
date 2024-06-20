# 混合专家网络

MoE 层：
- n 个“专家网络”：$E_1, \cdot, E_n$。
- 一个门控网络 $G$，其输出是一个稀疏的 $n$ 维向量。

给定输入 $x$，定义 $G(x)$是门控网络的输出；$E_i(x)$ 是第 $i$ 个专家网络的输出。于是 MoE 模块的输出为：
$$ y=\sum_{i=1}^{n} G_i(x) E_{i}(x) $$
基于 $G(x)$ 输出的稀疏性，可以节省计算量。当 $G_i(x)=0$时，我们无需计算 $E_i(x)$。如果专家数目非常大，可以采用层次化的 MoE：

主选通网络是 Gprimary，次选通网络为$(G_1, G_2, \cdots, G_a）$，专家网络为 $(E_{0,0}, E_{0,1}, \cdots, E_{a,b})$。MoE的输出为：
$$y_{H}=\sum_{i=1}^{a} \sum_{j=1}^{b} G_{primary, i}(x) \cdot G_{i, j}(x) \cdot E_{i, j}(x) $$


## 门控网络
1. Softmax Gating
用一个矩阵乘上输入，然后经过一个 Softmax 函数，这种方法实际上是一种非稀疏的门控函数：

$$ G_{\sigma}(x)=\operatorname{Softmax}\left(x \cdot W_{g}\right) $$

2. Noise Top-K Gating
在 Softmax 门控网络基础上加入两个元素：**稀疏性**和**噪声**。在执行 Softmax 函数之前：
   1) 在执行 Softmax 函数之前， 加入可调的高斯噪声，噪声项是为了帮助负载均衡（load balancing），并且保留前 k 个值，其他设置为 $-\infty$。 
   这种稀疏性是为了节省计算资源，尽管这种形式的稀疏性，从理论上会造成一些可怕的输出间断性，但在实际使用中，并没有观察到这种问题。每个分量的噪音量，通过另一个可训练的权重矩阵来控制。

$$ G(x)=\operatorname{Softmax}(\operatorname{KeepTopK}(H(x), k)) $$

$$ H(x){i}=\left(x \cdot W{g}\right){i}+ StandardNormal ()\cdot \operatorname{Softplus}\left(\left(x \cdot W{\text {noise }}\right)_{i}\right) $$

$$ KeepTopK (v, k){i}=
\left{\begin{array}{ll}v{i} & \text { if } v_{i} \text { is in the top } k \text { elements of } v \ -\infty & \text { otherwise. }\end{array}\right. $$
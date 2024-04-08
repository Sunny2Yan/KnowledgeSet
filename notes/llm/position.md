# Position Embedding

位置编码是在序列数据中为每个位置添加位置信息。在nlp中，位置编码通常用于处理文本序列，可以帮助模型更好地理解和处理序列数据。
Transformer中，通过为每个位置分配一个固定的向量来实现，其设计目的是使模型能够区分不同位置的输入。

## 1. 绝对位置编码

绝对位置编码是为序列数据中的每个位置添加绝对位置信息，即通过为每个位置分配一个唯一的向量来表示绝对位置信息。
由于绝对位置编码只关注单个位置信息，因此它的实现通常在输入层，可以通过简单的向量相加融入模型。

### 1.1 learned 编码 (bart,bert)

根据位置信息编码id，再通过训练Embedding层，学习该位置Embedding参数，这样每个位置都被映射为一个固定长度的向量。
这种位置编码具有一个较大的缺陷，即不具备外推性能，一旦在infer时的序列长度超过了train阶段的长度，它的位置信息就位于模型的盲区。

### 1.2 三角编码 (transformer)

$$
PE(pos, 2i) = sin(pos / 10000^{(2i/d_{model})}) \\
PE(pos, 2i+1) = cos(pos / 10000^{(2i/d_{model})})
$$

其中，pos表示位置，i表示维度，$d_{model}$表示模型隐藏层的维度。
通过使用不同频率的正弦和余弦函数，位置编码可以捕捉到不同位置之间的相对距离和顺序。

## 2. 相对位置编码

相对位置编码是为序列数据中的每个位置添加相对位置信息，为了让模型能够更好地理解序列中不同位置之间的相对关系和顺序。
相对位置是pair对的信息，因此无法直接在输入层实现，通常是通过改变Attention_score的计算方式来实现。即，在这种方法中，模型通过计算不同位置之间的相对位置偏移量，并将这些偏移量作为注意力机制的输入，以便模型能够更好地关注不同位置之间的相对关系。

### 2.1 ROPE (llama)

旋转位置编码（Rotation Position Encoding，RoPE）是通过引入旋转矩阵来表示位置之间的旋转关系，从而捕捉序列中位置之间的旋转模式。

假设通过某个运算 $f(\cdot,m), f(\cdot, n)$ 来给 $q, k$ 添加绝对位置信息：

$$
\tilde{q}_m=f(q, m);~~\tilde{k}_n=f(k, n)
$$

是的经过运算后 $\tilde{q}_m,\tilde{k}_n$ 带有位置 $m, n$ 的绝对位置信息。经过 Attention 的内积运算后带有相对位置信息：

$$
\langle f(q, m),~f(k, n)\rangle =g(q, k, m-n)
$$

且需要满足初始条件：$f(q, 0)=q; f(k, 0)=k$。

借助复数中 $\langle q, k\rangle=\mathrm{Re}(qk^H)$ 来求解，即 $\mathrm{Re}[f(q, m)f^{H}(k, n)]=g(q, k, m-n)$。对于二维形式的$q, k$都有：

$$
f(x_m, m)=
\left(
\begin{matrix} 
\cos m\theta & -\sin m\theta  \\
\sin m\theta & \cos m\theta
\end{matrix}
\right)

\left(
\begin{matrix} 
W_{11} & W_{12}  \\
W_{21} & W_{22}
\end{matrix}
\right)
\left(
\begin{matrix} 
x_1  \\
x_2
\end{matrix}
\right)
$$

即，相当于在 $(x_1, x_2)$ 处旋转了 $m\theta$ 度角到 ($x_1^{'}, x_2^{'}$)。而对于 d 维向量有：

$$
f(x_m, x) = R_{\theta, m}^{d}Wx_m  \\
R_{\theta, m}^{d} = 
\left(
\begin{matrix} 
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0  \\
\sin m\theta_1 &  \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0  \\
0 & 0& \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0  \\
0 & 0& \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0  \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots  \\
0 & 0 & \cdots & 0 & 0 & \cos m\theta_{d/2} & -\sin m\theta_{d/2}  \\
0 & 0 & \cdots & 0 & 0 & \sin m\theta_{d/2} &  \cos m\theta_{d/2}  
\end{matrix}
\right)
$$

其中，$\theta_i=10000^{-2(i-1)/d}, i\in [1, 2, \cdots, d/2]$。为了减少计算量，对 $R_{\theta, m}^{d}$ 做出变形：

$$
R_{\theta, m}^{d} = 
\left(
\begin{matrix} 
x_1 \\
x_2 \\
x_3 \\
x_4 \\
\vdots \\
x_{d-1} \\
x_d
\end{matrix}
\right)
\otimes

\left(
\begin{matrix} 
\cos m\theta_1 \\
\cos m\theta_1 \\
\cos m\theta_2 \\
\cos m\theta_2 \\
\vdots \\
\cos m\theta_{d/2} \\
\cos m\theta_{d/2}
\end{matrix}
\right)
+
\left(
\begin{matrix} 
-x_2 \\
x_1 \\
-x_4 \\
x_3 \\
\vdots \\
-x_{d} \\
x_{d-1}
\end{matrix}
\right)
\otimes

\left(
\begin{matrix} 
\sin m\theta_1 \\
\sin m\theta_1 \\
\sin m\theta_2 \\
\sin m\theta_2 \\
\vdots \\
\sin m\theta_{d/2} \\
\sin m\theta_{d/2}
\end{matrix}
\right)
$$

**RoPE的远程衰减特性**：

$$
\begin{aligned}
<q_m, k_n> &= Re[q_mk_n^H]  \\
&= Re[e^{i(m-n)\theta_0} + e^{i(m-n)\theta_1} + \cdots + e^{i(m-n)\theta_{d/2-1}}]  \\
&= Re[\sum_{k=0}^{d/2-1}e^{i(m-n)\theta_{-k/(d/2)}}]  \\
&\sim \frac{d}{2}\cdot Re[\int_0^1e^{i(m-n)\cdot 10000^{-t}}dt]
\end{aligned}
$$

**RoPE的优点**：

1. 通过引入旋转操作，可以解决周期性问题，使得位置编码可以适应更长的序列。
2. 可以捕捉到相对位置信息，使得模型能够更好地建模序列中的局部关系。
3. 对于处理不同长度的序列以及在测试时遇到未见过的序列长度，具有一定的外推性。

### 2.2 ALiBi (baichuan)

ALiBi（Attention with Linear Biases）是研究长度外推性的开山之作，通过引入线性偏置来改进自注意力机制（Self-Attention）。
ALiBi的思路是给Attention_score加上一个关于距离的线性偏置项 这样做的好处是，线性映射可以将输入序列的信息压缩到一个更紧凑的表示中，从而减少模型对长距离依赖关系的建模难度。

$$
a_{i,j}=q_ik_j^T-m|i-j|
$$

其中，m是与heads相关的超参数，对于n_heads=8的模型，它的取值分别为 $\frac{1}{2^1}, \frac{1}{2^2}, \cdots, \frac{1}{2^8}$。

**ALiBi中偏置矩阵的作用**：

- 用于调整注意力权重的矩阵。其中，偏置矩阵是一个形状为（L，L）的矩阵，其中L是输入序列的长度。
- 通过调整偏置矩阵的值，可以控制注意力权重的稀疏性和集中性，以更好地适应不同长度的序列。
- 通过调整注意力权重的分布，模型可以更好地适应不同长度的序列，并更好地捕捉序列中的长距离依赖关系，从而提高模型的长度外推性能。

**ALiBi的优点**：

1. 引入线性偏置来改进自注意力机制，增强了模型对长距离依赖关系的建模能力。这样可以更好地捕捉序列中的长距离依赖关系，提高模型的性能。
2. 灵活性：ALiBi中的偏置矩阵提供了一种灵活的方式来调整注意力权重的分布。通过调整偏置矩阵的值，可以控制注意力权重的稀疏性和集中性，以更好地适应不同长度的序列。这种灵活性使得ALiBi能够适应不同的任务和数据特点。
3. 减少参数数量：ALiBi使用线性映射将输入序列转换为一个低维度的特征向量，从而减少了模型的参数数量。这样可以降低模型的复杂度，减少计算和存储成本，并提高模型的效率。
4. 通用性：ALiBi可以应用于各种长度外推问题，如序列预测、机器翻译等。它的思路和方法可以适用于不同领域和任务，具有一定的通用性。

综上所述，ALiBi通过改进自注意力机制，提供了一种灵活的方式来调整注意力权重的分布，减少参数数量，并具有一定的通用性。这些优点使得ALiBi在处理长度外推问题时具有较好的性能和适应性。

## 3. 长度外推
长度外推问题是指在短序列上训练的模型，能否不用微调地用到长序列上并依然保持不错的效果。其主要包含两个问题”
1. 预测的时候用到了没训练过的位置编码；
2. 预测的时候注意力机制所处理的token数量远超训练时的数量

长度外推问题通常是由于训练数据的限制或资源限制而引起的，模型需要学会推断和生成超出其训练数据长度范围的内容。

### 3.1 长度外推问题的解决方法

1. 使用适当的模型架构：选择能够处理不同长度序列的模型架构。
2. 使用适当的位置编码方法：如 RoPE 可以外推10%到20%左右；
3. 增加训练数据范围：如果可能，可以增加训练数据的范围，包括更长的序列示例；
4. scale attention: $att(Q,K,V)=softmax(\frac{\log_m^n}{\sqrt{d}}QK^T)V$，其中m是训练长度，n是预测长度

## 4. 长度扩展
1. 直接外推
   模型对没被训练过的位置不具有适应能力，在推理阶段扩展长度，效果会不好。
2. position interpolation (内插)
   $$f'(x, m)=f(x, \frac{mL}{L'})$$
   即将新的长度按比例压缩到原来窗口内，压缩后更加“拥挤”，通常需要微调。
3. NTK (高频外推、低频内插)
   位置n的旋转位置编码本质上是$\beta$进制编码，即，RoPE的构造基础就是Sinusoidal位置编码：
   $$[cos(\frac{0}{\beta^0}), sin(\frac{1}{\beta^0}), cos(\frac{2}{\beta^1}), sin(\frac{3}{\beta^1}), \cdots, cos(\frac{n-1}{\beta^{d/2-1}}), sin(\frac{n}{\beta^{d/2-1}})], \beta=10000^{2/d}$$
   其中，最低频是 $\frac{n}{\beta^{d/2−1}}$ 项，引入参数 $\lambda$ 变为 $\frac{n}{(\beta\lambda)^{d/2−1}}$ ，使其与内插一致，即：
   $$\frac{n}{(\beta\lambda)^{d/2−1}} = \frac{n/k}{\beta^{d/2−1}}$$
   解得 $\lambda=k^{2/(d−2)}$，code直接修改base `base = base * 8 ** (dim / (dim-2))`
4. rerope
   按照高频外推、低频内插的思想，设定一个窗口大小w，在窗口内使用大小为1的位置间隔，在窗口外使用大小为1/k的位置间隔，即: 
   $$[0, 1, \cdots, w, w+\frac{1}{k}, w+\frac{2}{k}, \cdots, w+\frac{L-1-w}{k}]$$
   其中，需要w小于训练长度。当 $k\rightarrow \infty$ 时，有：
   $$[0, 1, \cdots, w, w, w, \cdots, w]$$

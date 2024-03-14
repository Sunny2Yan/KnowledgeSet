# Activation Function

## 1. 二分类激活函数
### 1.1 Sigmoid (LR)
$$Sigmig(x) = \frac{1}{1+e^{-x}}$$

Sigmoid极容易导致梯度消失问题，且计算比较费时。

### 1.2 Tanh ()
$$Tanh(x) = \frac{sinh(x)}{cosh(s)}=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

关于原点对称，但也极容易导致梯度消失问题，且计算比较费时。

## 2. ReLU (标准attention)
ReLU（Rectified Linear Unit）非线性整流单元

$$ReLU(x) = max(0, x)$$

## 3. SiLU (llama)

$$SiLU(x) = x * sigmoid(x)$$

## 3. GeLU (Bert, GPT2)

GeLU（Gaussian Error Linear Unit）高斯误差线性单元

$$GeLU(x) = 0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715x^3)))$$

GELU具有更平滑的非线性特征，这有助于提高模型的性能，并且能够加速模型收敛；但是，计算复杂度较高，可能会增加模型的计算开销。

## 4. Swish ()
Google 提出对ReLU的平替函数。

$$Swish(x) = x \cdot sigmoid(\beta \cdot x)$$

$\beta$ 为可训练参数，当 $\beta$ 为0时，Swish函数退化为线性函数；当 $\beta$ 趋近于无穷大时，Swish函数趋近于ReLU函数。Swish函数相对于其他激活函数来说计算开销较大，因为它需要进行Sigmoid运算。

## 5. GLU ()
GLU（Gated Linear Unit）线性门控单元，通过引入门控机制来增强模型的非线性能力。

$$GLU(x) = x \cdot sigmoid(W_1 \cdot x)$$

$W_1$ 是一个可学习的权重矩阵，Sigmoid 函数的输出称为门控向量，用来控制输入向量 x 的元素是否被激活。

GLU 能够对输入向量进行选择性地激活，从而增强模型的表达能力。但 GLU 的计算复杂度较高，可能会增加模型的计算开销。

### 5.1 使用 GeLU 激活函数的 GLU 块

$$GLU(x) = x \cdot GeLU(W_1 \cdot x)$$

使用 GeLU 作为 GLU 块的激活函数可以增强模型的非线性能力，提供更好的性能和更快的收敛速度。

### 5.2 使用 Swish 激活函数的 GLU 块

$$GLU(x) = x \cdot sigmoid(W_1 \cdot x)$$

使用Swish作为GLU块的激活函数可以增强模型的非线性能力，提供更好的性能和更快的收敛速度。

## 6. swiGLU (llama系列)

$$GLU(x,W_1,W_2,b,c)=\sigma(xW_1+b) \odot (xW_2+c) \\
swiGLU(x,W_1,W_2,b,c,\beta)=swish(xW_1+b) \odot (xW_2+c)$$

$\odot$ 为Hadamard积，表示逐位相乘。
# 量化 （Quantization）

## 1. 数据类型
在计算机科学中，数值通常由3部分构成：符号位（Sign）、指数部分（Exponent）、小数部分（Fraction）。
动态范围（dynamic range）：所表示的最小值与最大值的区间；
精度（precision）：相邻两个数值的间隔。
注：指数部分控制范围，小数部分控制精度。小数部分越多，数值精确越高。

- Float 32-bit: 1S-8E-23F  标准数据类型 4byte
- Float 16-bit: 1S-5E-10F  
- BFloat 16-bit: 1S-8E-7F  范围同fp32，但精度更低
- Int 8-bit: 1S-0E-7F      范围[-2^7-1, 2^-1]

$$
V = (-1)^{sign} * 2^{exponent-bias} * (1 + \frac{fraction}{2^F});  
$$

## 2. 量化方式
### 2.1 对称量化（symmetric quantization）
对称量化前后的值域都是围绕零点对称。典型的量化方法是**绝对最大值（absolute maximum）量化**，如下将 m-bit 映射到 n-bit:
$$
\begin{aligned}
scale\_factor &= \frac{2^{n-1} - 1}{\max(abs(x))}; \\
x_{quantized} &= round(scale\_factor \cdot x); \\
x_{dequantized} &= \frac{x_{quantized}}{scale\_factor}
\end{aligned}
$$

### 2.2 非对称量化（asymmetric quantization）
非对称量化不是以零为中心对称，如 int8[-128, 127]。如下将 m-bit 的区间 [a, b] 映射到 n-bit 的区间 [c, d]：
$$
\begin{aligned}
scale\_factor &= \frac{c - d}{b - a}; \\
zero\_point &= round(-scale\_factor \cdot a) - 2^{n-1}; \\
x_{quantized} &= round(scale\_factor \cdot x + zero\_point); \\
x_{dequantized} &= \frac{x_{quantized} - zero\_point}{scale\_factor}
\end{aligned}
$$

举例：将 fp32 类型的区间 [-700, 1000] 映射到 int8 类型的区间 [-128, 127]:
$$
s = \frac{127 - -128}{1000 - -700} = 0.15 \\
z = round(-0.15 \cdot -700) - 2^{8-1} = -23\\
x_{quantized} = round(0.15 \cdot x + -23) \\
x_{dequantized} = \frac{x_{quantized} - -23}{0.15}
$$

注：使用以上量化方式需要裁剪掉异常值（outlier）
**量化损失**：原始值与反量化值之间的差值，即 $x_{dequantized} - x$

### 2.3 校准过程（Calibration）
校准就是找到一个能够包含尽可能多数值的范围，同时尽量减少量化误差。

- 权重（静态值）：weights、biases
- 激活值（动态值）：推理过程中的输入值（或是经过激活函数后的输出值，也是下一层的输入）

针对权重值量化：
1. 读取权重范围，计算scale直接量化；

针对激活值量化：
1. 优化原始权重与量化后权重之间的均方误差（MSE）； 
2. 优化原始输出值与量化输出值之间的 KL 散度。

## 3. 量化方法
### 3.1 训练后动态量化（Post Training Quantization，PTQ dynamic）
1. 将训练好的模型权重量化为指定类型，并保存量化后的模型参数；
2. 对于每一层，将输入动态量化为指定类型； 再用量化后的权重和量化后的输入进行计算；
3. 将每一层的输出反量化为原始类型。

```python
### pytorch 将 fp32 量化为 int8 ###

#                int8_weight -|
#                             |-> int8 计算 -(dequant)-> fp32 
# fp32_input -(quant)-> int8 -|

import torch

class Model(torch.nn.Module):
    ...

model = Model()
quant_layers = {torch.nn.Linear, }
model_int8 = torch.ao.quantization.quantize_dynamic(model, quant_layers, dtype=torch.qint8)
```

### 3.2 训练后静态量化（PTQ static）
1. 将训练好的模型权重量化为指定类型，并保存量化后的模型参数；
2. 校准：利用具有代表性的数据进行模型推理，用数据在网络每一层产生的激活估算出激活值的量化参数；
3. 对于每一层，先做指定类型计算，再将结果反量化为原始类型，同时根据校准产生的激活值量化参数，把激活值量化为int8。

```python
# int8_weight -|
#              |-> int8 计算 -(dequant + quant)-> int8_output
# int8_input  -|

import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()  # 量化占位符
        self.linear_1 = torch.nn.Linear(10, 10, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(10, 2, bias=False)
        self.dequant = torch.ao.quantization.DeQuantStub()  # 反量化占位符
        
    def forward(self, x):
        q_inputs = self.quant(x)
        outputs = self.linear_2(self.relu(self.linear_1(q_inputs)))
        f_outputs = self.dequant(outputs)
        
        return f_outputs

def inference_loop(infer_model):
    ...

model = CustomModel()
model.qconfig = torch.ao.quantization.get_default_qconfig("x86")  # 适用于x86架构
model_prepared = torch.ao.quantization.prepare(model)  # qat = quantization aware train
inference_loop(model_prepared)  # 模型训练过程
model_int8 = torch.ao.quantization.convert(model_prepared)
```

## 量化感知训练（Quantization Aware Training，QAT）
边训练边量化（一般无损，经过训练metric会更好）。过程如下：
1. 加载fp32的模型权重，fp32的模型输入；
2. 在网络里插入模拟量化节点（fake_quantization）来分别对模型参数和激活值进行量化和反量化，从而引入量化误差；
3. 模型在fp32精度下进行训练。

```python
# training
# fp32_weight -(quant + dequant)-|
#                                |-> fp32 计算 -> fp32_output
# fp32_input  -(quant + dequant)-|

import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear_1 = torch.nn.Linear(10, 10, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(10, 2, bias=False)
        self.dequant = torch.ao.quantization.DeQuantStub()
        
    def forward(self, x):
        q_inputs = self.quant(x)
        outputs = self.linear_2(self.relu(self.linear_1(q_inputs)))
        f_outputs = self.dequant(outputs)
        
        return f_outputs

def train_loop(train_model):
    ...

model = CustomModel()
model.qconfig = torch.ao.quantization.get_default_qconfig("x86")  # 适用于x86架构
model_prepared = torch.ao.quantization.prepare_qat(model)  # qat = quantization aware train
train_loop(model_prepared)  # 模型训练过程
model_int8 = torch.ao.quantization.convert(model_prepared)
```

## 4. 大模型量化方法

### 4.1 GPTQ（完全在GPU上）
在训练后，逐层（layer-wise）进行非对称量化。

#### 4.1.1 最优脑损伤（Optimal Brain Damage，OBD）
一种剪枝方法，思想：参数的贡献度=删除该参数后，目标函数的变化量。
设模型的权重为 W，损失函数为 $L(\cdot)$，则在当前权重下，模型的训练损失为 L = L(W)。对其进行泰勒展开如下：
$$
\Delta L=(\frac{\partial L}{\partial W})^T\Delta W+\frac{1}{2}\Delta W^{T}H\Delta W+O(||\Delta W||^3)
$$

目标：找到一个参数集合，使得删除掉这个参数集合之后 $L$ 的增量最小。
由于目标函数中存在较大的 Hessian 矩阵，且 Hessian 矩阵的逆复杂度太高，不容易直接求解。故作如下近似：
1. 训练好的神经网络模型，处于权重空间中的局部极小值，因此认为 $\frac{\partial L}{\partial W} = 0$；
2. 对角逼近理论：删除多个参数所引起的 $\delta L$ 是单独删除每个参数所引起的 $\delta L$ 的和，因此 Hessian 矩阵为对角阵；
3. 对角近似：忽略 Hessian 矩阵中非对角线元素。

$$
\Delta L=\frac{1}{2}\Delta W^{T}H\Delta W \Longrightarrow \delta L=\frac{1}{2}\sum_{i}\delta w_i^2h_{ii}
$$

```text
1. 选择一个合理的网络架构
2. 训练网络直到收敛
3. 计算每个参数的二阶偏导数值 h_{kk}
4. 根据公式 s_{k}=h_{kk}\frac{w_k^2}{2}计算每个参数的贡献度
5. 将参数按照贡献度排序，并删除一些贡献度低的参数
6.迭代至第二步
```

#### 4.2 最优脑手术（Optimal Brain Surgeon，OBS）
主要思路同上，并假设对第 $q$ 个参数剪枝，即 $\delta w_q + w_q = 0$，于是转化为带约束的凸优化问题：
$$
\arg\min_{q}\frac{1}{2}\Delta W^{T}H\Delta W  \\
s.t.~~~~e_q^T\delta W + w_q = 0
$$
其中，$e_q$ 为 $q$ 处为 1 的单位向量。将上式转化为拉格朗日形式：
$$
\begin{aligned}
\delta L&=\frac{1}{2}\delta W^{T}H\delta W+\lambda(e_q^T \delta W+w_q)  \\
& \Longrightarrow \lambda = \frac{w_q}{[H^{-1}]_{qq}}; 注意 e_q^TH^{-1}e_q=[H^{-1}]_{qq} \\
& \Longrightarrow \delta W^T = -\frac{w_q}{[H^{-1}]_{qq}}e_q^TH^{-1} \\
\delta L &= \frac{w_q^2}{2 [H^{-1}]_{qq}} 
\end{aligned}
$$

```text
1. 训练一个相当大的神经网络至收敛;
2. 计算 Hessian 矩阵的逆矩阵 H^{-1};
3. 找到使得 L_q=\frac{1}{2}\frac{w_q^2}{[H^{-1}]_{qq}} 最小的 q。如果这个候选误差比 E 小得多，那么第 q 个权重就应该被删除，然后转到步骤 4，否则转到步骤 5。(也可以选择其他的停止条件);
4. 使用第 3 步的q，用公式 \delta w=-\frac{w_q}{[H^{-1}]_{qq}}*H^{-1}*e_q 更新所有的权重，然后转到步骤 2;
5. 没有更多的权重可以删除，结束。(可以进行重训练)
```

#### 4.3 最优脑压缩（Optimal Brain Compression，OBC）
OBS的每次迭代中都需要计算 Hessian 矩阵的逆，时间复杂度为 $O(n^4)$。

Row-wise 权重剪枝：
定义了 layerwise 的模型剪枝，即对每一层，定义剪枝损失：
$$
\delta L(f(X, W), f(X, W'))
$$
其中 $W, W'$分别为原始权重和剪枝后的权重矩阵。对于线性层、卷积层，可以表示为 $f(X, W) = WX$，于是有：
$$
\arg\min_{\hat{W}} ||WX - \hat{W}X||^2
$$
将权重矩阵按行分解，则剪枝损失可以表示为：
$$
\begin{aligned}
\delta L &= \sum_{i=1}^{d_{row}}\delta L_i = \sum_{i=1}^{d_{row}} ||W_{i,:}X - \hat{W}_{i,:}X||^2 \\
\delta L_i &= ||W_{i,:}X - \hat{W}_{i,:}X||^2 = \sum_{k=1}^{N}(\sum_{j=1}^{d_{col}}(w_{ij}-\hat{w}_{ij})x_{jk})^2
\end{aligned}
$$
由于每次迭代只对某一行中的一个权重进行剪枝，即只影响一行的剪枝损失，所以认为在一次迭代中 $\delta L = \delta L_i$。
$$
\begin{aligned}
H_{pq} &= \frac{\partial^2 \delta L_i}{\partial w_{ip} \partial w_{iq}} \\
&= 2\sum_{k=1}^{N} x_{pk}x_{qk} \\
\Longrightarrow H^{(i)} &= 2XX^T
\end{aligned}
$$

```text
1. 根据 XX^T 计算初始 Hessian 矩阵 H^{(i)}，并为权重矩阵的每一行复制一份；
2. 遍历每一行，根据 OBS 中 q 的求解公式 计算出最佳权重索引 q_i，其中 i 为所在行索引，对应的 Hessian 矩阵是 H^{(i)}；
3. 根据 OBS 中 \delta W^T 的求解公式计算当前行的权重修正量，然后更新权重矩阵；
4. 更新第 i 行的 Hessian 矩阵 H^{(i)}，并根据公式 (6) 更新 Hessian 逆矩阵；
5. 重复步骤 2-4，直到达到剪枝目标。
```
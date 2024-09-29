# torch int8 量化
为什么量化对神经网络影响不大
1. 模型经过Normalization，基本数值范围都不大；
2. 激活函数将数值影响平滑；
3. 对于分类器，只要最后概率值高于其他类别就可以

## 训练后动态量化
直接对训练好的模型进行量化，不需要其他操作。过程如下：
1. 将训练好的模型权重量化为int8，并保存量化参数；
2. 对于每一层，输入fp32，动态量化为int8； 再用量化后的int8权重和int8输入进行计算；
3. 将每一层的输出反量化为fp32

```python
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

## 训练后静态量化
对训练后的模型进行量化。过程如下：
1. 将训练好的模型权重量化为int8，并保存量化参数；
2. 校准：利用具有代表性的数据进行模型推理，用数据在网络每一层产生的激活估算出激活值的量化参数
3. 对于每一层，先做int8计算，再将结果反量化为fp32，同时根据校准产生的激活值量化参数，把激活值量化为int8

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

## 量化感知训练
边训练边量化（一般无损，经过训练metric会更好）。过程如下：
1. 加载fp32的模型权重，fp32的模型输入；
2. 在网络里插入模拟量化节点（fake_quantization）来分别对模型参数和激活值进行量化和反量化，从而引入量化误差；
3. 模型在fp32精度下进行训练。

```python
# fp32_weight -(quant + dequant)-|
#                                |-> fp32 计算 -> fp32_output
# int8_input  -(quant + dequant)-|

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

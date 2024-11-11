# 数据并行
针对训练数据集太大的问题：
1. 将数据集分为N份，每一份分别装载到N个GPU节点中，同时，每个GPU节点持有一个完整的模型副本；
2. 基于每个GPU中的数据分别去进行梯度求导；
3. 在GPU0上对每个GPU中的梯度进行累加，然后将GPU0聚合后的结果广播到其他GPU节点。

note 1: 上面是以GPU0作为参数服务器，此外，还可以使用CPU作为参数服务器。但是训练速度通常会慢于使用GPU0作为参数服务器（通常情况下，GPU与CPU之间通信使用PCIe，而GPU与GPU之间通信使用Nvlink）

note 2: 可以将参数服务器分布在所有GPU节点上面，每个GPU只更新其中一部分梯度（每个GPU都需要聚合全部梯度）。

note 3: 数据并行不仅仅指对训练的数据并行操作，还可以对网络模型梯度、权重参数、优化器状态等数据进行并行。

## 1. Data Parallel
1. 将 inputs(batch), model 从主 GPU 分发到所有 GPU 上 (sub_batch); 
2. 每个 GPU 分别独立进行前向传播，得到 outputs;
3. 将每个 GPU 的 outputs 发回主 GPU, 通过 loss function 计算出 loss，并求导，得到损失梯度，分发到所有 GPU 上；
4. 各 GPU 反向传播计算参数梯度，并将所有梯度回传到主 GPU，通过梯度更新模型权重。
5. 不断重复上面的过程。

```python
import torch

model = torch.nn.Linear(10, 10).to('cuda_programming')
input_var = torch.randn(20, 10).to('cuda_programming')
net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
output = net(input_var)  # input_var can be on any device, including CPU
```

缺点：
1. 只能在单机多卡上使用，不支持分布式；
2. DataParallel使用单进程多线程实现的，受困于 GIL，会带来性能开销，速度很慢；
3. 主卡性能和通信开销容易成为瓶颈，GPU 利用率通常很低；
4. 不支持模型并行，也没办法与模型并行组合使用。

## 2. Distributed Data Parallel (此时 DP 已被弃用)
基于上述问题，分布式数据并行基于多进程实现，每个进程都有独立的优化器，执行自己的更新过程。每个进程都执行相同的任务，且都与所有其他进程通信。进程（GPU）之间只传递梯度，这样网络通信就不再是瓶颈。

1. 将 rank=0 进程中的模型参数广播到进程组中的其他进程；
2. 每个 DDP 进程都会创建一个 local Reducer 来负责梯度同步； 
3. 在训练过程中，每个进程从磁盘加载 batch 数据，并传递到 GPU，每个 GPU 都有自己的前向过程，完成前向传播后，梯度在各个 GPUs 间进行 All-Reduce，每个 GPU 都收到其他 GPU 的梯度，从而可以独自进行反向传播和参数更新； 
4. 同时，每一层的梯度不依赖于前一层，所以梯度的 All-Reduce 和后向过程同时计算，以进一步缓解网络瓶颈； 
5. 在后向过程的最后，每个节点都得到了平均梯度，这样各个 GPU 中的模型参数保持同步

缺点：
1. 需要整个模型加载到一个GPU上；

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def example(rank, world_size):
    # 创建进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # 构建DDP模型
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    optimizer.step()


if __name__=="__main__":
    # 设置环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2  # GPU数
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)
```

## 3. DeepSpeed ZeRO
在模型训练的过程中，GPU上需要进行存储的参数包括了模型参数、优化器状态、激活函数的输出值、梯度以及一些零时的Buffer。
在进行混合精度运算时，其中模型状态参数(优化器状态 + 梯度+ 模型参数）占据主要。因此，可以想办法去除模型训练过程中的冗余数据。
其中，模型参数2P、梯度参数2P、优化器参数12P（Adam类型优化器需要存储模型、动量momentum、动量二阶矩variance）

ZeRO对 模型状态（Model States）参数进行不同程度的分割，主要有三个不同级别：

ZeRO-1: 对优化器状态分片（Optimizer States Sharding）
ZeRO-2: 对优化器状态和梯度分片（Optimizer States & Gradients Sharding）
ZeRO-3: 对优化器状态、梯度分片以及模型权重参数分片（Optimizer States & Gradients & Parameters Sharding）

1. forward过程由每个rank的GPU独自完整的完成，然后进行backward过程。在backward过程中，梯度通过allReduce进行同步； 
2. Optimizer state 使用贪心策略基于参数量进行分片，以此确保每个rank几乎拥有相同大小的优化器内存；
3. 每个rank只负责更新当前优化器分片的部分，由于每个rank只有分片的优化器state，所以当前rank忽略其余的state；
4. 在更新过后，通过广播或者allGather的方式确保所有的rank都收到最新更新过后的模型参数。

## 4. Fully Sharded Data Parallel
完全分片数据并行是Pytorch最新的数据并行方案，主要用于训练大模型。其思想借助于ZeRO


```python
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import default_auto_wrap_policy

model = nn.Linear(8, 4).to('cuda_programming')
model = DistributedDataParallel(model())

fsdp_model = FullyShardedDataParallel(
    model(), 
    fsdp_auto_wrap_policy=default_auto_wrap_policy, 
    cpu_offload=CPUOffload(offload_params=True),
)

optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
for sample, label in next_batch():
    out = fsdp_model(input)
    loss = criterion(out, label)
    loss.backward()
    optim.step()
```

手动包装（Manual Wrapping）:
可以有选择地对模型的某些部分应用包装，总体设置可以传递给enable_wrap()上下文管理器
```python
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import enable_wrap, wrap

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = wrap(nn.Linear(8, 4))  # 只包装某些层
        self.layer2 = nn.Linear(4, 16)
        self.layer3 = wrap(nn.Linear(16, 4))
 
wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
    fsdp_model = wrap(model())

optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
for sample, label in next_batch():
    out = fsdp_model(input)
    loss = criterion(out, label)
    loss.backward()
    optim.step()
```
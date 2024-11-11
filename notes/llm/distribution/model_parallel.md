# 模型并行

在数据并行训练中，每个 GPU 都持有整个模型权重的副本，这就带来了冗余问题。模型并行是将模型分割并分布在一个设备阵列上，每一个设备只保存模型的一部分参数。

模型并行分为张量并行和流水线并行：

- 流水线并行：层间并行，对模型不同的网络层进行分割；
- 张量并行：层内并行，对模型的每一个网络层进行分割；

## 1. pipeline parallel
1. 流水线并行策略
   1) F-then-B 模式：先进行前向计算，结束后再进行反向计算。
      缺点：需要缓存多个 micro-batch 的中间变量和梯度，显存的实际利用率并不高。 
   2) 1F1B 模式：前向计算和反向计算交叉进行的方式，即 micro_batch_1 前向执行完立即执行反向计算，在执行 micro_batch_2 前向...

2. 朴素流水线并行
   将模型按照层间切分成多个 Stage，并将每个 Stage 分配给一个 GPU。然后对小批量数据进行常规的训练，在模型 Stage 的边界处进行通信。
   ```python
   output=L4(L3(L2(L1(input))))
   intermediate=L2(L1(input))
   output=L4(L3(intermediate))
   ```
   缺点：每次只有一个 GPU 在运行，存在很多bubble，计算效率太低；且存在设备之间复制数据的通信开销。

3. MicroBatch 流水线并行
   与朴素流水线几乎相同，但它通过将传入的小批次（minibatch）分块为微批次（microbatch），并人为创建流水线来解决 GPU 空闲问题；
   如：GPU1 执行完 micro_batch_1 发送给 GPU2 执行，此时 GPU1 执行 micro_batch_2。
   缺点：对于需要统计量的层（如：Batch Normalization），会导致计算麻烦

4. GPipe（Easy Scaling with Micro-Batch Pipeline Parallelism）
   Gpipe 流水线并行主要用来解决这两个问题：
   1) 提高模型训练的并行度: 将 mini-batch 的数据细分为多个更小的 micro-batch（在训练时计算和运用的是micro-batch里的均值和方差）；
   2) 采用激活重计算（Re-materialization）降低显存消耗：前向传播时，不用计算激活的结果，在反向传播时重新计算。

   ```python
   # Need to initialize RPC framework first.
   os.environ['MASTER_ADDR'] = 'localhost'
   os.environ['MASTER_PORT'] = '29500'
   torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
   
   fc1 = nn.Linear(16, 8).cuda(0)
   fc2 = nn.Linear(8, 4).cuda(1)
   model = nn.Sequential(fc1, fc2)
   
   from torch.distributed.pipeline.sync import Pipe
   # chunks表示micro-batches的大小，默认值为1
   model = Pipe(model, chunks=8)
   input = torch.rand(16, 16).cuda(0)
   output_rref = model(input)
   ```
   
5. PipeDream (deepspeed 团队)
    PipeDream 采用 1F1B 策略，使得 activation 的缓存数量只跟 stage 数相关，从而节省显存。
   
## 2. tensor parallel
张量并行是将计算图中的层内的参数切分到不同设备（即层内并行），每个设备只拥有模型的一部分，以减少内存负荷。
按照张量切分方式分为行并行与列并行：
   - Row Parallelism：按模型参数 W 的行分割；
      X*W = [X1, X2]*[W1, W2]^T = X1W1+X2W2 = Y1+Y2 = Y
   - Column Parallelism：按模型参数 W 的列分割：
      X*W = X*[[W1], [W2]] = [XW1, XW2] = [Y1, Y2] = Y

```python
import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module

device_mesh = DeviceMesh("cuda_programming", torch.arange(0, world_size))

model = Model().cuda(rank)
optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

# 给定的并行风格为PairwiseParallel，将 colwise 和 rowwise 样式串联为固定对。
model = parallelize_module(model, device_mesh, PairwiseParallel())

for i in range(iter_nums):
    # 对于 TP，所有 TP rank 的输入需要相同。
    # 设置随机种子是为了模仿数据加载器的行为。
    if rank==0:
        print(f"-----------{i}--------------")
    torch.manual_seed(i)
    inp = torch.rand(20, 10).cuda(rank)
    if rank==0:
        print(f"rank: {rank} , input shape: {inp.shape}")
    output = model(inp)
    if rank==0:
        print(f"rank: {rank} , input shape: {output.shape}")
    output.sum().backward()
    optimizer.step()
```

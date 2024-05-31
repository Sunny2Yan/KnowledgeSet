# 显存问题

### 1. 模型参数量计算
对于 transformer 模型：
vocab_size=V; hidden_size=H; intermediate_size(mlp_hidden_size)=H'; layers=L

Embedding(VH) + L * [ATT(3HH + HH) + MLP(2HH' + H'H) + Norm(H + H)] + Output(HV)

## 2. 模型存储大小
1B = 10 亿参数；按半精度 16bytes 计算，每个参数占 2B；即 $10^9 * 2B$
换算成GB：$10^9 * 2 / 1024 / 1024 / 1024 \approx 1.8 GB$ 

推理模型需要的显存约等于模型文件大小，全参训练需要的显存约为推理所需显存的三倍到四倍。

## 3. 模型训练显存大小
模型训练显存占用包含：模型参数、模型梯度、优化器参数；
目前大模型训练方案采用混合精度训练：模型参数与梯度参数是 fp16，优化器参数是 fp32 

设模型参数为 P：
显存大小 = 2P(模型) + 2P(梯度) + 4P(优化器：模型、动量、动量二阶矩) * 3 = 16P

## 4. 模型运算量计算 （浮点运算次数 FloatingPoint Operations, FLOP）
对于 transformer 模型：
model_parameter=P; batch_size=B; seq_len=S
训练词元总数𝐶=𝐵S; num_header=𝑁，header_dim=𝐷，中间状态维度𝐻=𝑁𝐷

已知矩阵乘积运算计算量为：[m,n] * [n, p] = 2mnp

1. multi_head_att: 
   $Q, K, V \in \R^{B \times S \times H}$; 多头计算时需要拆分: $Q', K', V' \in \R^{B \times N \times S \times D}$
    Q'K'^T = 2(BNSD * BNDS) =  2BNSDS = 2BS^2ND
    缩放(元素级操作): BNS^2; Softmax: 3BNS^2(指数，加和，归一化，都是元素级操作); 乘V': 2BS^2ND

   一次multi_head_att: (4BS^2ND + 4BNS^2) * L = 4BS^2N(D+1)L = 4CSL(H+D)
   前向+反向：3 * 4BS^2N(D+1) = 12BS^2N(D+1)  （transformer中反向传播计算量约为前向的两倍）

2. 线性变换：
   multi_head_att中四个线性层: 2BSH * HH * 4 = 8BSHH = 8CHH
   MLP中三个线性层: 2BSH * HH’ * 3 = 6BSHH‘ = 6CHH'
   (注：若训练时采用激活重计算技术，需要额外进行一次前向传播，即2BSH * HH’ * 4 = 8CHH')
   输出层一个: 2BSH * HV = 2BSHV = 2CHV

参数量为𝑃 的模型在 𝐶 个词元上进行预训练的总体运算量 $\approx 6CP$ 

## 5. 模型训练时间
$$训练时间 = \frac{模型运算量}{GPU数量 * GPU每秒浮点运算数}$$

## 6. 显卡利用率评估
1. flops比值法： 
   gpu利用率 = 实测的flops / 显卡理论上的峰值flops
2. throughout估计法：
   吞吐量 = example数量/秒/GPU * max_length
   gpu利用率 = 实际吞吐量 / 论文中的吞吐量（假设利用率100%）
3. torch profiler分析法：
   利用torch profiler记录各个函数的时间，将结果在tensorboard上展示，在gpu kenel视图下，可以看到tensor core的利用率

## 7. 其他指令
```bash
iftop -i eth2 -n  -P  # 查看多机训练时的网速
nvidia-smi topo -m  # 查看服务器上的多卡之间的NVLINK topo
ds_report  # 查看对 deepspeed 的环境配置是否正确
```

基于deepspeed训练，可以通过配置文件查看训练时的 flops
```
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
    }
}
```
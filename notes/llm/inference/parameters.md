# Inference Parameters

## 搜索方式
目的：为了增加模型输出的多样性。
- Random Sampling: 随机采样；
- Greedy Search: 贪心搜索，每次都选取概率最大的token；
- Beam Search: 束搜索，每次保留beam_size个概率最大的token，形成一个树结构
- Nuclear Sampling: 核采样，核采样：每次保留累积概率达到 p 的tokens，形成一个树结构

## 参数
1. top-k: 针对 beam search，每次选取 k 个概率最大的 tokens，然后对其随机采样；
2. top-p: 针对 nuclear sampling，每次选取累积概率达到 p 的 tokens，然后随机采样（弥补了top-k中的k不好取值问题）；
3. temperature: 将输出logits 除以 T，即 $softmax(y_{i})=\frac{e^{y_{i}}}{\sum_{j=1}^{n} e^{y_{j}}} \rightarrow softmax(y_{i})=\frac{e^{\frac{y_{i}}{T}}}{\sum_{j=1}^{n} e^{\frac{y_{j}}{T}}}$；
   $T \rightarrow \infty$: 随机采样；$T \rightarrow 0$: top-1采样。
4. repetition_penalty: 重复惩罚，
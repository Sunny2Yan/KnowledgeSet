# llm interview

## 基础
1. cross entropy (用来度量两个概率分布间的差异)
   $H(p, q) = -\sum_{x} p(x) \log(q(x))$; 交叉熵刻画了两个概率分布之间的距离，值越小，两个概率分布越接近
   对于二分类：$L=\frac{1}{N}\sum_{i}L_i =- \frac{1}{N} \sum_{i}[y_i \log(p_i)]$; y表示真实分布
   对于多分类：$L=\frac{1}{N}\sum_{i}L_i =- \frac{1}{N} \sum_{i} \sum_{c=1}^{M}y_{ic}\log(p_{ic})$; $y_{ic}$取0或1，样本i的类别等于c取1

KL散度理解？  
其中对数概率的作用？
Recall，Precision的计算
场景题，在处理数据的时候面对有违规的语料（如黄暴内容）如何平衡Recall，Precision指标
大模型灾难遗忘怎么解决
如何避免模型过拟合
如何解决大模型遗忘问题

## 分词
tokenizer 的分词方法
   [tokenization](notes/llm/tokenizer.md)
## 模型结构
1. attention
   $att=softmax(\frac{qk^T}{\sqrt{d}}) v$
   初始的Attention很接近one hot分布，不除以根号d，会造成梯度消失
   multi_head：可以学习到不同的知识，增强表达能力
2. transformer 结构、位置编码
   单向transformer模型:
   pre-training:
   fine-tuning:
   Loss: CrossEntropyLoss 交叉熵损失

   结构：(n_layers=6, att_head=8, hidden=512, mlp_hidden=4*hidden, seq_len=256)
   tokenization: SentencePiece + BPE
   position: $PE(pos, 2i)=sin(pos / 10000^{2i / d}); PE(pos, 2i+1)=cos(pos / 10000^{2i / d})$
   embedding: token_embed + position_embed
   activation: ReLU
   normalization: layernorm ($w * (x - \hat{x}) / (s + \epsilon) + b$

   6 * EncoderBlock (=> norm(x + dropout(multi_head_att(x))) -> norm(x + dropout(mlp(x))))  后norm ==> en_out
   6 * DecoderBlock (=> norm(x + dropout(multi_head_att(x))) -> norm(x + dropout(cross_att(x, en_out, en_out))) -> norm(x + dropout(mlp(x))))  后norm -> linear
   multi_head_att: x -> q, k, v -> att -> linear
   cross_multi_head_att: x -> q, en_out = k, en_out = v -> att -> linear

3. Bert (only-encoder)
   双向transformer模型：$P(w_i | w_1, \cdots, w_{i-1}, w_{i+1}, \cdots, w_n)$
   pre-training: (task_1: mask_lm(随机mask 15%并预测，vocab类)；
                  task_2: next_sentence_predict(输入AB两个句子，判断B是不是A的下一句))
   fine-Tuning: 分类(输入AB，判断两个句子是否具有相关性)
   Loss：Negative Log Likelihood 负对数似然损失 $-\sum_1^n{\log{p(x_i; \theta)}}$
   
   结构：(n_layer=12, att_head=12, hidden=768, mlp_hidden=4*hidden, dropout=0.1, seq_len=512)
   tokenization: WordPiece
   position: $PE(pos, 2i)=sin(pos / 10000^{2i / d}); PE(pos, 2i+1)=cos(pos / 10000^{2i / d})$
   embedding: token_embed + position_embed + segment_embed(句子拼接: [cls]A[sep]B[sep] ==> 000111)
   activation: $GELU(x)=x/2 *(1 + tanh(\sqrt{(2 / \pi)}*(x + 0.44715x^3)) )$
   normalization: layernorm ($w * (x - \hat{x}) / (s + \epsilon) + b$
   
   12 * EncoderBlock (=> x + multi_head_att(norm(x)) -> x + mlp(norm(x)) -> dropout)  先norm
   multi_head_att: x -> q, k, v -> att -> linear

4. GPT (only-decoder)
   单向transformer模型：$P(w_i | w_{i-k}, \cdots, w_{i-1})$
   pre-training: 根据第一个token预测后面的token; LMHead: linear(vocab)
   fine-tuning: n分类问题; ClsHead: linear(vocab) -> linear(n)
   Loss: CrossEntropyLoss 交叉熵损失 (fine-tune时，cls_loss + ratio * lm_loss, 防止下游精调时出现灾难性遗忘问题)
   
   结构：(n_layers=12, att_head=12, hidden=768, mlp_hidden=4*hidden, dropout=0.1, seq_len=512)
   tokenization: SentencePiece (gpt1); BPE (gpt2)
   position: nn.Embedding(0-512) (gpt1)
   embedding: token_embed + position_embed
   activation: $GELU(x)=x/2 *(1 + tanh(\sqrt{(2 / \pi)}*(x + 0.44715x^3)) )$
   normalization: layernorm ($w * (x - \hat{x}) / (s + \epsilon) + b$

   12 * DecoderBlock (=> norm(x + dropout(multi_head_att(x))) -> norm(x + dropout(mlp(x))))  gpt2先norm
   mask_multi_head_att: x -> q, k, v -> mask_att -> linear
   mask_att: att_score = mask(att_score)

   gpt2中引入了past_key_value, 防止模型在文本生成任务中重新计算上一次迭代计算好的上下文值；
   gpt3中引入了稀疏注意力机制和自适应注意力跨度来提高计算效率和长距离依赖的建模能力

5. llama结构、rmsnorm、激活函数swiGLU、position embedding构造方法
   单向transformer模型：$P(w_i | w_{i-k}, \cdots, w_{i-1})$
   pre-training: 根据前面的token预测后一个token， temperature > 0时，softmax(logits/temperature)并采样top_p
   fine-tuning: sft, instruction-tuning
   Loss: CrossEntropyLoss 交叉熵损失

   结构：(n_layers=32, att_head=32, hidden=4096, mlp_hidden=4*hidden, seq_len=2048)  llama2: seq_len=4096
   tokenization: SentencePiece + BPE
   position: RoPE [旋转位置编码](notes/llm/position.md)
   embedding: token_embed + position_embed
   activation: $SiLU(x) = x * sigmoid(x)$
   normalization: RMSNorm ($W * \frac{x}{\sqrt{\frac{1}{n} \sum_i^n{x_i^2} + \epsilon}}$)

   32 * DecoderBlock (=> x + multi_head_att(norm(x)) -> x + mlp(norm(x)))  先norm
   mask_multi_head_att: x -> q, k, v -> rope(q, k) -> mask_att -> linear
   mask_att: att_score = mask(att_score)

## peft
1. prompt tuning (sft)
   hard prompt: 类似于in-context-learning中的few shot.
   soft prompt: 把 Prompt 的生成作为一个任务进行学习，相当于把人工设计离散的 Prompt 变成模型自己进行学习、尝试（连续）

   Prompt Tuning: 训练一个PromptEmbedding层，将人工输入或随机的prompt template调整为模型能够理解的 prompt token。 
   流程：frozen llm, token_embedding = prompt_embedding + text_embedding (原始模型的embedding输出)

2. prefix tuning (sft)

3. p-tuning
   
4. lora

5. qlora

6. Ptuning和全量微调对比
7. 介绍lora，p-turing，各自优缺点

## 训练
1. sft
训练数据量级？
训练方法，用的什么sft，有什么不同，有什么优缺点，原理上解释不不同方法的差别
模型微调会性能下降为什么还需要这一步？
prompt tuning, instruct tuning, fine tuning差别
如何关注训练过程中的指标？ 训练步数如何确定？
评估指标是什么，这些指标存在哪些问题

2. rlhf
马尔科夫决策过程的定义，有哪些参数变量需要考虑？
有了解隐马尔科夫链吗，细说(给出公式那种)
CRF

RLHF流程、优化目标公式、
目标公式中衰减因子的作用，取大取小有什么影响？
RLHF的目标公式可以加入什么其他的项？
熵正则项是如何加入的？ 

为什么需要Rewar model？
Reward model 如何训练？Reward model 你觉得训练到什么程度可以？
Reward model不准确怎么办？
Rewar model和训练的LLM模型用同一个基座模型可能有什么作用？
Reward有多个目标可以怎么做？
Reward model 训练的loss是什么？
RLHF中PPO算比率相对什么来算？
为啥RLHF中要用PPO？和其他RL算法的区别？
DPO、DPO的原理？

```
RM（Reward Model，奖励模型）的数据格式
输入数据是一个句子，奖励数据是一个实数值，表示对输入数据的评价。每一行代表一个样本，第一列是输入数据，第二列是对应的奖励数据
Input,Reward
"This is a sentence.",0.8
"Another sentence.",0.2
```

## 推理
模型推理是怎么做的，有没有cot，tot等等，还是单轮

## rag
langchain中的每个模块都了解么；
如何根据本地知识库实现对话；（rag）

## 问题1：context length
如何解决content length长度问题

## 问题二： 幻觉
模型为什么会胡言乱语？根源在哪？可以考虑如何解决？
大模型主要存在的问题和解决思路（幻觉，定向编辑，继续训练等等）

## 问题三：加速
大模型加速框架了解多少，知不知道原理 如何进行加速优化
deepspeed框架介绍
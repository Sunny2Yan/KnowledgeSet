# llm interview

## 基础
cross entropy公式
交叉熵损失函数写一下，物理意义是什么
KL散度理解？  
其中对数概率的作用？
Recall，Precision的计算
场景题，在处理数据的时候面对有违规的语料（如黄暴内容）如何平衡Recall，Precision指标
大模型灾难遗忘怎么解决
如何避免模型过拟合
如何解决大模型遗忘问题

## 分词
tokenizer 的分词方法

## 模型结构
1. attention (多头：可以学习到不同的知识，增强表达能力)
2. transformer 结构、位置编码
3. Bert结构、输入、输出、预训练任务、特点等等
4. gpt结构、损失函数、gpt1-3的区别
5. llama结构、rmsnorm、激活函数swiGLU、position embedding构造方法

为什么计算注意力 QK 分数要除以维度开根号  (scaling后进行softmax操作可以使得输入的数据的分布变得更好，数值会进入敏感区间，防止梯度消失，让模型能够更容易训练。)
位置嵌入

## peft
1. LORA的理解 
2. Ptuning和全量微调对比
3. 介绍lora，p-turing，各自优缺点

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
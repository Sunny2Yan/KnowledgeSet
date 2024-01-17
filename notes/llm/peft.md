# Parameter-efficient Fine-tuning

## 1. Prompt Learning
提示学习能够通过在输入中添加一个提示词（Prompt），使得预训练模型的性能大幅提高。

**动机：**
1. 语言模型越来越大，Fine-tune 的成本也越来越高，且预训练模型越大，fine-tune需要的数据也越多；
2. 较大的语言模型一般支持多任务，使用 Fine-tuning 需要重新多次训练预训练模型，导致占用较高的内存。

In-context Learning: task description + few shot sample + question。

**Prompt 分类：**
Hard Prompt (Discrete Prompt)：
认为设计的 Prompt（上面的in-context learning），一般需要模型在这个域上有比较多的经验；

Soft Prompt (Continuous Prompt)：
把 Prompt 的生成作为一个任务进行学习。相当于把 Prompt 的生成从人类一个一个尝试（离散）变换成机器自己进行学习、尝试（连续）。软提示不可避免地往模型内引入了新的参数，这就引入了一个新的问题：**如何参数有效地学习软提示**？（P-Tuning、Prefix-Tuning...）

## 2. Prompt Tuning (P-Tuning)
### 2.1 标准的 Prompt Tuning
在预训练语言模型的输入中添加可学习的 embedding 向量作为提示。这些提示被设计成在训练过程中更新，以引导模型输出对特定任务更有用的响应.

### 2.2 P-Tuning (Prompt-based Tuning)
Prompt tuning 是使用可训练的虚拟token embedding，P-Tuning 则是使用一个可训练的LSTM模型（称为prompt_encoder）来动态生成虚拟token embedding。


# 3. Prefix Tuning
动机：
传统的fine tuning范式利用预训练模型去对不同的下游任务进行微调：
1. 一方面微调整个模型耗时长；
2. 对每个任务都需要保存一份微调后的模型权重，会占很多存储空间。

基于上述两点，Prefix Tuning提出固定预训练LM，为LM添加可训练，任务特定的前缀，这样就只需要为不同任务保存不同的前缀，微调成本也降低。

优势：
1. 不需要调整模型的全部权重，只是通过在输入中添加前缀来调整模型的行为，可以节省大量的计算资源；
2. 一个单一的模型能够适应多种不同的任务，前缀可以是固定的（hard prompt）或可训练的（soft prompt）。

Prefix Tuning 与 Prompt Tuning 的区别：
Prompt Tuning：倾向于用少量的 embedding 模仿传统的自然语言提示；
Prefix Tuning：前缀作为模型内部表示的一部分，可以影响整个模型的行为。

## 4. Adapter Tuning
目标是在不改变预训练模型的原始参数的前提下，使模型能够适应新的任务。

方法：
在模型的每个层或某些特定层之间插入小的神经网络模块，称为“adapters”。在模型 fine tuning 时保持原始模型参数不变，只调整adapters。

## 5. LoRA (Low-Rank Adaptation)
方法：
将模型的关键层的权重矩阵分解为两个低秩矩阵的乘积，fine tuning时只更新这两个低秩矩阵，最后将乘积加到原始权重矩阵上，这样不会直接改变整个模型的结构。
矩阵通常位于模型的多头自注意力和前馈神经网络部分。

优势：
在不增加太多额外计算负担的情况下，有效调整模型，同时保持其原有的性能。

Adapter Tuning和LoRA的区别：
Adapter Tuning：通过在模型的各个层中添加“adapters”，来实现微调。这些适配器独立于模型的主体结构，只有它们的参数在微调过程中被更新，而模型的其他预训练参数保持不变。
LoRA：通过在模型的权重矩阵中引入低秩矩阵（通常是两个小的矩阵的乘积）来实现对模型的微调。这些低秩矩阵作为原有权重矩阵的修改项，使得原有的权重矩阵在实际计算时得到调整。

## 6. QLoRA (Quantized Low-Rank Adaptation)
在 LoRA 的基础上引入了深度量化过程。即，在训练过程中将模型用4-bit加载，然后在训练时把数值反量化到bf16后进行训练。这样的设计使得训练所需的显存大大减少。
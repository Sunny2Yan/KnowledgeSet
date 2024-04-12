# llm interview

## 基础
1. cross entropy (用来度量两个概率分布间的差异)
   $H(p, q) = -\sum_{x} p(x) \log(q(x))$; 交叉熵刻画了两个概率分布之间的距离，值越小，两个概率分布越接近
   对于二分类：$L=\frac{1}{N}\sum_{i}L_i =- \frac{1}{N} \sum_{i}[y_i \log(p_i)]$; y表示真实分布
   对于多分类：$L=\frac{1}{N}\sum_{i}L_i =- \frac{1}{N} \sum_{i} \sum_{c=1}^{M}y_{ic}\log(p_{ic})$; $y_{ic}$取0或1，样本i的类别等于c取1

2. kl divergence
   $L(y_{pre}, y_{ture}) = y_{true} \log{\frac{y_{ture}}{y_{pre}}} = y_{true}(\log{y_{true}} - \log{y_{pre}})$

3. Precision, Recall


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
   初始化：任务相关的实体文本进行tokenize来初始化 （10-20 token）

2. prefix tuning (sft)
   传统的fine-tuning花费较大，且不同的下游任务需要存储不同的模型，prefix只需要保存prefix layer即可

   流程：添加一个prefix(embed+mlp或直接embed)，自回归模型表示为 [prefix;x;y]; encoder-decoder模型表示为 [prefix;x;prefix';y]
   初始化：Embedding(num_virtual_tokens, token_dim) 没有实际意义

3. p-tuning
   针对encoder-decoder模型，添加 MLP(LSTM(input_embed)) 模块
   
5. lora
   llm在预训练后，越大的模型权重矩阵的秩越小，于是将需要fine-tune的参数矩阵W变成两个小矩阵的乘积 W=AB，即：$W_0+\Delta W=W_0 +AB$

   流程：
   初始化：A（高斯分布），B（初始化为0）

## 训练
### sft
训练数据量级：llama(1T); llama2(2T)
训练步数: 一般3个epoch
评估指标: ，这些指标存在哪些问题

### rlhf
1. 马尔科夫决策过程:
马尔可夫决策过程是一个4元组 $(S,A,P_{a},R_{a})$，其中：
   - S是状态空间的集合
   - A是动作的集合
   - $P_{a}(s,s')=P(s_{t+1}=s'\mid s_{t}=s,a_{t}=a)$ 是 t 时刻 s 状态下的动作 a 导致 t+1 时刻进入状态 s' 的概率
   - $R_{a}(s,s')$ 状态 s 经过动作 a 转换到状态 s' 后收到的即时奖励（或预期的即时奖励）
   - 策略函数 $\pi$ 是从状态空间 S 到动作空间 A 的映射。

有了解隐马尔科夫链吗，细说(给出公式那种)
CRF

2. [RLHF流程](notes/llm/rlhf.md)
   policy: GPT; action_space: 全词表; observation_space: 全词表*seq_len; reward;

   step 1: query_tensor -> sft_model -> response_tensor
   step 2: query_tensor + response_tensor -> reward_model(小) -> reward
   step 3: 
      ```
      q_a -> reward_model            -> score            -> +
      q_a -> actor_model             -> logits_1 -|
                                                  |-> kl(logits_1 || logits_2)
      q_a -> critic_model(ref_model) -> logits_2 -|
      ```
   优化目标: $r=r_{\theta} - \lambda r_{KL}$

目标公式中衰减因子的作用，取大取小有什么影响？
RLHF的目标公式可以加入什么其他的项？
熵正则项是如何加入的？ 
RLHF中PPO算比率相对什么来算？
为啥RLHF中要用PPO？和其他RL算法的区别？

3. Reward model
   数据格式：RewardDataCollatorWithPadding
   ```
      input_ids_chosen: "question + good_answer"
      attention_mask_chosen`
      input_ids_rejected: "question + bad_answer"
      attention_mask_rejected
   ```
   model: 类型(SEQ_CLS)属于Text classification (1分类，即打分)
   loss: $-LogSigmoid(x) = -\log{(\frac{1}{1+e^{-x}})}$, 即 `-nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()`
   trick：使用多个奖励模型的输出，增加数据度量的信息源
   Reward多目标：？？？

4. 近端策略优化 (Proximal Policy Optimization, PPO)
   ```
   初始化policy参数 $\theta_0$ 和惩罚项权值 $\beta_0$，kl-divergence $delta$
   for $k = 0, 1, 2, \cdots$ do:
   在policy $\pi_k=\pi(\theta_k)$ 上收集一批经验数据 $D = {(s, a, r, s')}$
   对于 $D$ 中的每个经验 $(s, a, r, s')$ do:
      使用任意的优势评估算法评估优势 $\hat{A_t^{\pi_k}}$
      计算policy更新：$\theta_k+1 = \arg\max_{\theta} L_{\theta_k}(\theta) - \beta_k D_{kl}(\theta || \theta_k)$
      if $D_{kl}(\theta_k+1 || \theta_k) \leq 1.5\delta$ :
         $\beta_{k+1} = 2 \beta$
      elif $D_{kl}(\theta_k+1 || \theta_k) \geq delta/1.5$:
         $\beta_{k+1} = \beta / 2$ 
   end for   
   ```
   
5. DPO
PPO、DPO的原理？


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
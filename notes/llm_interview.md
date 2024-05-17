# llm interview

## 基础

1. cross entropy (用来度量两个概率分布间的差异)
   $H(p, q) = -\sum_{x} p(x) \log(q(x))$; 交叉熵刻画了两个概率分布之间的距离，值越小，两个概率分布越接近
   对于二分类：$L=\frac{1}{N}\sum_{i}L_i =- \frac{1}{N} \sum_{i}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$; y表示真实分布
   对于多分类：$L=\frac{1}{N}\sum_{i}L_i =- \frac{1}{N} \sum_{i} \sum_{c=1}^{M}y_{ic}\log(p_{ic})$; $y_{ic}$取0或1，样本i的类别等于c取1
2. kl divergence
   $L(y_{pre}, y_{ture}) = y_{true} \log{\frac{y_{ture}}{y_{pre}}} = y_{true}(\log{y_{true}} - \log{y_{pre}})$
3. Precision, Recall
   TP(True Positive), TN(True Negative), FP(False Positive), FN(False Negative)
   Precision(准确率) = TP/(TP+FP)
   Recall(召回率) = TP/(TP+FN)
   F1-score = 2 * Precision * Recall / (Precision + Recall)
4. 大模型灾难遗忘问题
   定义：大模型学习新知识的过程会迅速破坏之前获得的信息，而导致模型性能在旧任务中急剧下降。

如何避免模型过拟合

## 分词

tokenizer 的分词方法
[tokenization](../notes/llm/tokenizer.md)

## 模型结构

1. attention
   $att=softmax(\frac{qk^T}{\sqrt{d}}) v$
   初始的Attention很接近one hot分布，不除以根号d，会造成梯度消失
   multi_head_att：可以学习到不同的知识，增强表达能力
   
   设序列长度为 l，注意力的计算复杂度为 O(l^2); 为了降低复杂度可以使用稀疏注意力机制，复杂度为 O(wL)
   稀疏注意力机制：在计算注意力时，使用滑动窗口注意力机制（Sliding Window Attention, SWA）,窗口为 w。

   ```python
   batch_size, seq_len, d_model = 1, 512, 768
   x = torch.rand((batch_size, seq_len, d_model))
   query, key, value = nn.Linear(d_model, 3 * d_model)(x).chunk(3, dim=2)  # 创建qkv三个不同的矩阵 [1, 512, 768]
   attn_score = torch.matmul(query, key.transpose(-1, -2)) / (d_model ** 0.5)
   attn_weights = nn.Softmax(dim=-1)(attn_score)
   att = torch.matmul(attn_weights, value)
   output = nn.Linear(d_model, d_model)(att)
   ```

   在推理时，由于是预测下一个token，则下一个step的输入就包含了上一个step的内容，只是末尾多了一个token。
   那么下一个step的计算也应该包含上一个step的计算，于是 KV_Cache=[(k_0,v_0), (k_1,v_1), ...]。
   对于输入长度为 S，层数为 L，hidden_size为 d 的模型，需要缓存的参数量为：$2*s*d*L$
   以llama 7B为例：L=32, hidden_size=4096, s=1k时：2*1024*4096*32=268435456 (512M)

   ```python
   # 1. multi head attention (KV Cache 以空间换时间)
   # 输入分别经过W_q、W_k、W_v的变换之后，都切成了 num_head 份，维度也从 d_model 降到了d_head，再分别对每个head进行attention计算并拼接
   qkw_1 = nn.Linear(d_model, 3*d_model)
   q_1, k_1, v_1 = qkv(x).chunk(3, dim=2)

   # 2. multi query attention (减少空间消耗，降低性能)
   # 经过W_q、W_k、W_v的变换后只对 Q 进行切分，而 K、V直接在线性变换的时候把维度降到了d_head，然后这n个Query头分别和同一份K、V进行attention计算，之后把结果拼接起来。
   qkw_2 = nn.Linear(d_model, d_model + 2*d_head)  # d_model = num_head * d_head
   q_2, k_2, v_2 = qkv(x).split([d_model, d_head, d_head], dim=2)

   # 3. Group Query Attention (折中方案 llama2,3)
   # 经过W_q、W_k、W_v的变换后 Q 仍不变，而 K、V在线性变换的时候把维度降到了group*d_head，同一个group内的 Q 共享同一套K、V，不同group的Q所用的K、V不同
   group_head = num_head / group
   qkw_3 = nn.Linear(d_model, d_model + 2*group_head*d_head)
   q_3, k_3, v_3 = qkv(x).split([d_model, group_head*d_head, group_head*d_head], dim=2)
   ```
3. transformer (encoder-decoder)
   Loss: CrossEntropyLoss 交叉熵损失

   结构：(n_layers=6, att_head=8, hidden=512, mlp_hidden=4*hidden, seq_len=256)
   tokenization: SentencePiece + BPE
   position: $PE(pos, 2i)=sin(pos / 10000^{2i / d}); PE(pos, 2i+1)=cos(pos / 10000^{2i / d})$
   embedding: token_embed + position_embed
   activation: ReLU
   normalization: layernorm $w * (x - \bar{x}) / (s + \epsilon) + b$

   6 * EncoderBlock (=> norm(x + dropout(multi_head_att(x))) -> norm(x + dropout(mlp(x))))  后norm ==> en_out
   6 * DecoderBlock (=> norm(x + dropout(multi_head_att(x))) -> norm(x + dropout(cross_att(x, en_out, en_out))) -> norm(x + dropout(mlp(x))))  后norm -> linear
   multi_head_att: x -> q, k, v -> att -> linear
   cross_multi_head_att: x -> q, en_out = k, en_out = v -> att -> linear
4. Bert (only-encoder)
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
5. GPT (only-decoder)
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
6. Llama (only-decoder)
   单向transformer模型：$P(w_i | w_{i-k}, \cdots, w_{i-1})$
   pre-training: 根据前面的token预测后一个token， temperature > 0时，softmax(logits/temperature)并采样top_p
   fine-tuning: sft, instruction-tuning
   Loss: CrossEntropyLoss 交叉熵损失

   结构：(n_layers=32, att_head=32, hidden=4096, mlp_hidden=4*hidden, seq_len=2048)  llama2: seq_len=4096
   tokenization: SentencePiece + BPE
   position: RoPE [旋转位置编码](../notes/llm/position.md)
   embedding: token_embed + position_embed
   activation: $SiLU(x) = x * sigmoid(x)$
   normalization: (Root Mean Square) RMSNorm ($W * \frac{x}{\sqrt{\frac{1}{n} \sum_i^n{x_i^2} + \epsilon}}$)

   32 * DecoderBlock (=> x + multi_head_att(norm(x)) -> x + mlp(norm(x)))  先norm
   mask_multi_head_att: x -> q, k, v -> rope(q, k) -> mask_att -> linear
   mask_att: att_score = mask(att_score)

embedding-model:
bce-embedding (开源)
text-davinci (openai)

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
4. lora
   llm在预训练后，越大的模型权重矩阵的秩越小，于是将需要fine-tune的参数矩阵W变成两个小矩阵的乘积 W=AB，即：$W_0+\Delta W=W_0 +AB$

   流程：
   初始化：A（高斯分布），B（初始化为0）

## 训练

### sft

训练数据量级：llama(1T); llama2(2T); llama3(15T)
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

2. [RLHF流程](../notes/llm/rlhf.md)
   policy: GPT; action_space: 全词表; observation_space: 全词表*seq_len; reward;

   step 1: query_tensor -> sft_model -> response_tensor
   step 2: query_tensor + response_tensor -> reward_model(小) -> reward
   step 3:

   ```
   q_a -> reward_model(freeze)  -> score                          -| 
   q_a -> actor_model           -> log_probs -    |                +         -> PPO
                                                  |-> kl(log_probs || ref_log_probs)
   q_a -> ref_model(freeze)     -> ref_log_probs -|
   ```

   ref_model是冻结的sft_model，其目的是防止actor训歪。

   优化目标:

目标公式中衰减因子的作用，取大取小有什么影响？
RLHF的目标公式可以加入什么其他的项？
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
4. PPO (Proximal Policy Optimization, 近端策略优化)

   初始化policy参数 $\theta_0$ 和惩罚项权值 $\beta_0$，kl-divergence $delta$
   for $k = 0, 1, 2, \cdots$ do:
   $\;\;\;\;$ 在policy $\pi_k=\pi(\theta_k)$ 上收集一批经验数据 $D_k$
   $\;\;\;\;$ 使用任意的优势评估算法评估优势 $\hat{A_t^{\pi_k}}$
   $\;\;\;\;$ 通过执行 K 步minibatch来计算policy更新： $\theta_{k+1} = \arg\max_{\theta} L_{\theta_k}(\theta) - \beta_k D_{kl}(\theta || \theta_k)$
   $\;\;\;\;$ if $D_{kl}(\theta_{k+1} || \theta_k) \leq 1.5\delta$ :
   $\;\;\;\;$ $\;\;\;\;$ $\beta_{k+1} = 2 \beta$
   $\;\;\;\;$ elif $D_{kl}(\theta_{k+1} || \theta_k) \geq \delta/1.5$:
   $\;\;\;\;$ $\;\;\;\;$ $\beta_{k+1} = \beta / 2$
5. DPO

## Prompt Engineering

1. Prompt Creator (提示词生成器)

   假设你是一个prompt export，我想让chatgpt用python代码实现一个计算器，请给我一个好的prompt。
2. Structured Prompt：角色 + 任务 + 要求 + 提示

   角色：假设你是一个有着丰富经验的python程序员。
   任务：请用python代码绘制一个五角星。
   要求：请使用matplotlib这个库，线条使用红色。
   提示：五角星需要先计算五个顶点，然后在间隔一个顶点的两个顶点之间两两进行连线。
3. One / Few Shot Prompt

   将英语翻译为汉语：
   big => 大
   small =>
4. COT (Chain of Thought)

   one-shot cot:
   Q: Roger有5个网球。他又买了两罐网球，每个罐子有3个网球。他现在有多少个网球?
   A: Roger一开始有5个球。2罐3个网球，每罐等于6个网球。5 + 6 = 11。答案是11。
   Q: 餐厅有23个苹果。如果他们使用了20个苹果做午餐，又买了6个，他们还有多少个苹果?

   zero-shot cot:
   餐厅有23个苹果。如果他们使用了20个苹果做午餐，又买了6个，他们还有多少个苹果?

   让我们一步步思考 / 让我们逐步解决这个问题，以确保我们得到正确的答案(优先)。
   (Let's think step by step / Let's work this out in a step by step way to be sure we have the right answer.)
5. ReACT (Reason+Act 协同思考和动作)

   一种reinforce language agents，按照 think -> act -> observation -> think... 的模式来解决问题。其中，act就是和环境交互(如：查询互联网，调用工具，执行代码等)。

   prompt：尽你所能回答以下问题。您可以访问以下工具:\n\n{tools}\n\n使用以下格式:\n\nQuestion: 您必须回答的输入问题\nThought: 你应该经常思考要做什么\nAction: 要采取的行动，应该是 [{tool_names}] 中之一\nAction Input: 动作的输入\nObservation: 动作的结果\n... (其中 Thought/Action/Action Input/Observation 可以重复N次)\nThought: 我现在知道最后的答案了\nFinal Answer: 原始输入问题的最终答案\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}
6. Reflexion (失败后自我反思)

   一种reinforce language agents，按照 task -> trajectory -> evaluation -> Reflection(如果失败则反思) -> next trajectory... 的模式来解决问题。

## rag

langchain中的模块:
chains, prompts, models, indexes, memory, agents

1. chains: 链式pipeline和文档链，文档链如下：
   stuff: 将所有文档组成一个文档列表，全部放到context中（适用于小文档）；
   refine: 循环遍历每一个文档，每次输入中间答案（上一个文档的答案）和一个文档作为context（适用于大文档）；
   map reduce: 循环遍历每一个文档得到输出结果，将所有结果组合成新文档作为输入；
   map re-rank: 循环遍历每一个文档得到输出结果，并给出每个答案的确定性得分，返回得分最高的一个。
2. prompts: prompt templates
3. models: llms, chats, text_embedding
4. indexes: document_loaders, text_splitters, vectorstore, retrievers
   multi query retriever: 根据query生成多个问题，并根据这些问题检索相关文档；
   contextual compression: 压缩单个文档，避免返回不必要的内容；
   ensemble retriever: 使用多个retriever，根据算法对结果进行排序，返回更好的结果；
   multi vector retriever: 在大段文档中分割小段文档，检索小段文档并定位到大段文档；
   parent document retriever: 检索时，先获取小块文档，并根据它查找父文档 ID，并返回那些较大的文档；
5. memory: 内存管理，MessageHistory， buffer, KGMemory(知识图谱)...
6. agents: llm agent, multi agent ...

rag采用Top-k进行召回，这样存在检索出来的chunks不一定完全和上下文相关，最后导致大模型输出结果不佳。
rerank: 将原有的Top-k召回，扩大召回数量，在引入粗排模型（policy，小模型，LLM），对召回结果结合上下文进行重排，来改进rag效果。

langchain实现rag:

```python
# step 1: load pdf
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")
pages = loader.load_and_split()

# step 2: split text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size = 500,
   chunk_overlap = 50,)
docs = text_splitter.split_documents(pages)

# step 3: build vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS  # local: faiss; online: pinecone

embed_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
   documents=docs, embedding=embed_model , collection_name="openai_embed")

# step 4: retrieval 
query = "How large is the baichuan2 vocabulary?"
results = vectorstore.similarity_search(query, k=3)

# step 5: build prompt and model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
source_knowledge = "\n".join([x.page_content for x in results])
augmented_prompt = f"""Using the contexts below, answer the query.
   contexts: {source_knowledge}
   query: {query}"""
messages = [
   SystemMessage(content="You are a helpful assistant."),
   HumanMessage(content=augmented_prompt), ]
chat = ChatOpenAI(
   openai_api_key="",
   model='gpt-3.5-turbo')
res = chat(messages)
```

## 问题1：[context length](../notes/llm/position.md)

解决content length长度问题

## 问题二： 幻觉

定义：大模型回答不准确、前后不一致等问题，生成的内容并非基于训练数据或不符合事实。

原因：

1. 数据质量：训练数据的质量不足，噪声较多，会导致出现幻觉；或是某一类数据大量重复导致模型产生偏好；
2. 解码过程中的随机性：top-k(beam search), top-p(核采样), temperature(logits/T)；
   核采样：由于top-k中的k不好确定，top-p只从累积概率达到p的最小单词集合中选择一个单词
   eg: 0.664, 0.199, 0.105...， p=0.9时只从前两个采样
   temperature: 温度越小差异越大，温度越大差异越小
   使用的先后顺序是 top-k -> top-p -> Temperature
3. 最大似然性目标：大模型的训练目标是最大化下一个token的概率，因此，模型更看重看起来正确，而不是输出内容的正确性；
4. 训练数据中本身不包含问题相关内容（in-context learning 解决）；
5. 上下文理解：大模型需要理解上下文信息来生成准确的答案，如果上下文窗口长度不足或模型无法有效处理上下文信息，就会导致模型无法理解上下文含义，从而产生幻觉。

解决方法：

1. 提高数据质量（包括预训练数据和sft数据）；
2. 采用更长更好的位置编码；
3. prompt工程：采用更合理的prompt（如：cot），或agent（如：ReAct）或要求大模型不确定的不回答；
4. RAG借助外部知识，严格按照给定知识回答；
5. 集成学习：将多个模型的预测结果进行集成，以提高预测的准确性和鲁棒性。

## 问题三：加速

训练加速：[deepspeed](../notes/llm/deepspeed.md)
推理加速：

1. FlashAttention：通过矩阵分块计算以及减少内存读写次数的方式，提高注意力分数的计算效率；
2. PagedAttention：针对增量解码阶段，对于 KV 缓存进行分块存储，并优化了计算方式，增大了并行计算度，从而提高了计算效率；
3. TGI (Text Generation Inference)

## 问题四：国产大模型

阿里：通义千问
百度：文心一言
腾讯：hunyuan
科大讯飞：讯飞星火
字节：云雀
百川智能：baichuan
智谱AI：GLM
月之暗面：kimichat
稀宇科技: minimax (moe)
幻方： deepseek

## 问题五：测评
MMLU：综合性的大规模评测数据集。(单选，四选一，[问题：xxx; 选项：xxx；答案：A])；
BIG-Bench：综合评测体系。(QA，单选)；
HELM：一个全面而系统 的评估体系。（QA，单选）
C-Eval：一个专门为中文大语言模型设计的综合评测体系。(单选)
CMMLU、AGIEval、MMCU、M3KE和Xiezhi等。 
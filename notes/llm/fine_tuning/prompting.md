# Prompting

hard prompt: 类似于in-context-learning中的few shot.
soft prompt: 把 Prompt 的生成作为一个任务进行学习，相当于把人工设计离散的 Prompt 变成模型自己进行学习、尝试（连续）

1. prefix tuning (sft)
   传统的fine-tuning花费较大，且不同的下游任务需要存储不同的模型，prefix只需要保存prefix layer即可

   流程：对模型每一层都添加一个prefix(embed+mlp或直接embed)，自回归模型表示为 [prefix;x;y]; encoder-decoder模型表示为 [prefix;x;prefix';y]
   初始化：Embedding(num_virtual_tokens, token_dim) 没有实际意义
   注：为了防止直接更新Prefix的参数导致训练不稳定和性能下降的情况，在Prefix层前面加了MLP结构，训练完成后，只保留Prefix的参数。

2. prompt tuning (sft)
   Prompt Tuning: 仅在输入层训练一个PromptEmbedding层，将人工输入或随机的prompt template调整为模型能够理解的 prompt token。
   流程：frozen llm, token_embedding = prompt_embedding + text_embedding (原始模型的embedding输出)
   初始化：任务相关的实体文本进行tokenize来初始化 （10-20 token）

3. p-tuning
   将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理
   针对encoder-decoder模型，添加 MLP(LSTM(input_embed)) 模块
   注：virtual token的位置是可选的，不一定是前缀。目的是把传统人工设计模版中的真实token替换成可微的virtual token。

4. p-tuning v2
   在每一层都加入了Prompts tokens作为输入，而不是仅仅加在输入层，具体做法基本同Prefix Tuning。这样做有两个好处：
   - 更多可学习的参数（从P-tuning和Prompt Tuning的0.01%增加到0.1%-3%），同时也足够参数高效。 
   - 加入到更深层结构中的Prompt能给模型预测带来更直接的影响
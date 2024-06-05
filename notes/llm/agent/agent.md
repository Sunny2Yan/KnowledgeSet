# Agent

## Prompt 工程
1. Prompt Creator (提示词生成器)

   假设你是一个prompt export，我想让chatgpt用python代码实现一个计算器，请给我一个好的prompt。
2. Structured Prompt：角色 + 任务 + 要求 + 提示
   （角色 + 角色技能 + 任务关键词 + 任务目标 + 任务背景 + 任务范围 + 任务解决与否的判定 + 任务限制条件 + 输出格式 + 输出量）

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

## Agent 框架

### LangChain
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

### LlamaIndex

### MetaGPT

### TaskWeaver: (具有SOP 能力)
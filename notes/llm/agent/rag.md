# Retrieval Augmented Generation
检索增强: 给 LLM 提供外部数据库，对于用户问题 (Query)，通过一些信息检索 (Information Retrieval, IR) 的技术，先从外部数据库中检索出和用户问题相关的信息，然后让 LLM 结合这些相关信息来生成结果.

## 
解决的问题：
1. 长尾知识：对于一些非通用和大众的知识，LLM 通常不能生成比较准确的结果；
2. 私有数据：训练数据中不包含私域知识，可以使用 rag 解决；
3. 数据新鲜度：LLM 在预训练中使用的数据容易过时；
4. 来源验证和可解释性：LLM 生成的输出不会给出其来源，解释性差。

关键模块：
1. 数据获取
2. 文档分块: 按句子分割、按token数分割（一般512 tokens），等；
3. 数据索引：
   1) 链式索引: 通过链表的结构对文本块进行顺序索引;
   2) 树索引: 将一组节点 ( 文本块 ) 构建成具有层级的树状索引结构，每个父节点都是子节点的摘要；
   3) 关键词表索引：从每个节点中提取关键词，构建了每个关键词到相应节点的多对多映射；
   4) 向量索引（常用）：将文本块映射成一个固定长度的向量，然后存储在向量数据库中，检索时匹配向量相似度。
4. 查询与检索

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


## langchain中的细节:

1. 文档链：
   stuff: 将所有文档组成一个文档列表，全部放到context中（适用于小文档）；
   refine: 循环遍历每一个文档，每次输入中间答案（上一个文档的答案）和一个文档作为context（适用于大文档）；
   map reduce: 循环遍历每一个文档得到输出结果，将所有结果组合成新文档作为输入；
   map re-rank: 循环遍历每一个文档得到输出结果，并给出每个答案的确定性得分，返回得分最高的一个。
2. 文档切分方式：
   Character: 基于字符数拆分；
   Recursive Character: (建议使用)按不同的字符递归的分割，优先级为['\n\n', '\n', '']
   tokens: 按 token 数拆分；
   Semantic Chunker: 按句子拆分，几个句子一组。
3. prompts: prompt templates
4. models: llms, chats, text_embedding
5. indexes: document_loaders, text_splitters, vectorstore, retrievers
   multi query retriever: 根据query生成多个问题，并根据这些问题检索相关文档；
   contextual compression: 压缩单个文档，避免返回不必要的内容；
   ensemble retriever: 使用多个retriever，根据算法对结果进行排序，返回更好的结果；
   multi vector retriever: 在大段文档中分割小段文档，检索小段文档并定位到大段文档；
   parent document retriever: 检索时，先获取小块文档，并根据它查找父文档 ID，并返回那些较大的文档；
6. memory: 内存管理，MessageHistory， buffer, KGMemory(知识图谱)...
7. agents: llm agent, multi agent ...

rag采用Top-k进行召回，这样存在检索出来的chunks不一定完全和上下文相关，最后导致大模型输出结果不佳。
re-rank: 将原有的Top-k召回，扩大召回数量，在引入粗排模型（policy，小模型，LLM），对召回结果结合上下文进行重排，来改进rag效果。
         1) 单独训练一个排序模型；2) 使用llm+prompt进行排序；3) 使用llm生成数据训练排序模型。

## 流程

```
knowledge -(1.split:xxx)-> documents -(2.embedding_model)-> embeddings -|
                                                                        |
query -(embedding_model)-> embedding --> retriever    <-- vector_database
                                             |(3.) top-k
                                             |(4.re-rank) top-k
                                            prompt (query+context) --> llm --> answer
```
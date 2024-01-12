# -*- coding: utf-8 -*-

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate)
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain


class Memory:
    """存储历史的交互信息
    存储状态：message list;
    查询状态：
    """
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(api_key=openai_api_key, temperature=0)
        self.embed = OpenAIEmbeddings(openai_api_key=openai_api_key)

    @staticmethod
    def _sample_use_memory():
        # 方案一：
        # ConversationBufferWindowMemory(k=1)  # 最后k次交互
        # ConversationEntityMemory(llm=llm)  # 记录对话中的实体和有关实体的给定事实
        # ConversationKGMemory(llm=llm)  # 使用知识图谱来记录事实
        # ConversationSummaryMemory(llm=llm)  # 记录对话的摘要

        memory = ConversationBufferMemory(memory_key="chat_history",
                                          return_messages=True)  # Message格式
        memory.chat_memory.add_user_message("hi!")
        memory.chat_memory.add_ai_message("what's up?")
        # memory.save_context({"Human": "hi"}, {}...)  # 直接存入字符串

        # 方案二：(轻量级, 等价上面)
        history = ChatMessageHistory()
        history.add_user_message("hi!")
        history.add_ai_message("whats up?")

        # {'chat_history': "Human: hi!\nAI: what's up?"}
        return memory.load_memory_variables({}), history.messages

    def chat_memory(self, query: str):
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                MessagesPlaceholder(variable_name="chat_history"),  # memory
                HumanMessagePromptTemplate.from_template("{question}"), ])
        memory = ConversationBufferMemory(memory_key="chat_history",
                                          return_messages=True)
        conversation = LLMChain(llm=self.llm, prompt=prompt,
                                verbose=True, memory=memory)
        response = conversation({"question": query})

        return response

    def multi_input_chain_memory(self, text_dir: str, query: str):
        with open(text_dir, 'r', encoding='utf-8') as f:
            document = f.read()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(document)
        docsearch = FAISS.from_texts(
            texts, self.embed,
            metadatas=[{"source": i} for i in range(len(texts))], )
        docs = docsearch.similarity_search(query)

        template = """You are a chatbot having a conversation with a human.

        Given the following extracted parts of a long document and a question, create a final answer.

        {context}

        {chat_history}
        Human: {human_input}
        Chatbot:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"],
            template=template)
        memory = ConversationBufferMemory(memory_key="chat_history",
                                          input_key="human_input")
        chain = load_qa_chain(self.llm, chain_type="stuff",
                              memory=memory, prompt=prompt)

        chain({"input_documents": docs, "human_input": query},
              return_only_outputs=True)
        response = chain({"input_documents": docs, "human_input": query},
                         return_only_outputs=True)

        return response, chain.memory.buffer


if __name__ == '__main__':
    api_key = "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"
    query = "hi!"
    memory = Memory(api_key)
    # print(memory._sample_use_memory())
    print(memory.chat_memory(query))

    text_dir = "C:\\Users\\duyanb.XINAO\\Desktop\\mem.txt"
    query = "锅炉怎么点火？"
    print(memory.multi_input_chain_memory(text_dir, query))

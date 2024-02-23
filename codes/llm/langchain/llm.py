# -*- coding: utf-8 -*-
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType, initialize_agent, load_tools


# 设置环境变量导入api_key
# os.environ['OPENAI_API_KEY'] = ""
# llm = ChatOpenAI()
class ChatChain:
    """LangChain 的基本使用方法 (invoke, ainvoke, stream, astream, batch, abatch)"""
    def __init__(self, openai_key: str, model_name: str):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are world class technical documentation writer."),
            ("user", "{input}")])
        self.llm = ChatOpenAI(openai_api_key=openai_key,
                              model_name=model_name,
                              temperature=0,
                              verbose=True, )

    def chat(self, query):
        # self.llm.invoke("how can langsmith help with testing?")  # context
        chain = self.prompt | self.llm | StrOutputParser()  # prompt到llm再到解析结果的链
        return chain.invoke({"input": query})  # string

    def stream(self, query):
        for chunk in self.llm.stream(query):
            print(chunk.content, end="", flush=True)

    def check_cost(self, ):
        """检查大模型 tokens 的花费情况"""
        with get_openai_callback() as cb:
            result = self.llm.invoke("Tell me a joke")
            return cb

        # Agent 中大模型的花费
        # tools = load_tools(["serpapi", "llm-math"], llm=self.llm)
        # agent = initialize_agent(tools, self.llm,
        #                          agent=AgentType.OPENAI_FUNCTIONS,
        #                          verbose=True)
        # with get_openai_callback() as cb:
        #     response = agent.run("Tell me a joke")
        #     return cb


if __name__ == '__main__':
    api_key = ""
    model_name = "gpt-3.5-turbo-1106"
    chat_chain = ChatChain(api_key, model_name)
    chat_chain.stream("Write me a song about sparkling water.")

    # # 设置缓存，减少重复问题的花费
    # from langchain.globals import set_llm_cache
    # from langchain.cache import InMemoryCache
    # from langchain.cache import SQLiteCache
    #
    # set_llm_cache(InMemoryCache())
    # # set_llm_cache(SQLiteCache(database_path=".langchain.db"))  # SQLite 缓存
    # time_1 = time.time()
    # chat_chain.chat("Write me a song about sparkling water.")
    # print(time.time() - time_1)  # 9.373593091964722
    # time_2 = time.time()
    # chat_chain.chat("Write me a song about sparkling water.")
    # print(time.time() - time_2)  # 0.013096332550048828

# -*- coding: utf-8 -*-
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


if __name__ == '__main__':
    api_key = ""
    model_name = "gpt-3.5-turbo-1106"
    chat_chain = ChatChain(api_key, model_name)
    chat_chain.stream("Write me a song about sparkling water.")
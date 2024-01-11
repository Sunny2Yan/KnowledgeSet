# -*- coding: utf-8 -*-

from langchain.tools.tavily_search import TavilySearchResults

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class Agent:
    def __init__(self, openai_api_key: str):
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(api_key=openai_api_key,
                              model="gpt-3.5-turbo-1106", temperature=0)

    def retrieval_tool(self, text_dir: str):
        pages = PyPDFLoader(text_dir).load()
        document = ' '.join([page.page_content for page in pages])
        raw_documents = Document(page_content=document,
                                 metadata={'source': text_dir})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, )
        documents = text_splitter.split_documents([raw_documents])

        vector = FAISS.from_documents(documents, self.embeddings_model)
        retriever = vector.as_retriever()
        # docs = retriever.get_relevant_documents(query=query)
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith.", )

        return retriever_tool

    # @staticmethod
    # def search_tool():
    #     search = TavilySearchResults()  # 搜索引擎（tool）
    #
    #     return search

    def tool(self, text_dir: str, query: str):
        # search_tool = self.search_tool()
        retrieval_tool = self.retrieval_tool(text_dir)
        tools = [retrieval_tool]  # [search_tool, retrieval_tool]

        prompt = hub.pull(
            "hwchase17/openai-functions-agent")  # 不同的agent有不同的prompt
        # create_react_agent(); create_self_ask_with_search_agent();
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools,
                                       verbose=True, max_iterations=2)

        # stateless agent
        response_1 = agent_executor.invoke({"input": query})

        # state agent
        message_history = ChatMessageHistory()   # 在内存中记录会话历史
        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: message_history,
            input_messages_key="input",
            history_messages_key="chat_history", )
        response_2 = agent_with_chat_history.invoke(
            {"input": "hi!"},
            config={"configurable": {"session_id": "<foo>"}}, )

        response_3 = agent_with_chat_history.invoke(  # response带有历史信息
            {"input": query},
            config={"configurable": {"session_id": "<foo>"}}, )

        return response_3

    def stream_agent(self, text_dir: str, query: str):
        retrieval_tool = self.retrieval_tool(text_dir)
        tools = [retrieval_tool]

        prompt = hub.pull(
            "hwchase17/openai-functions-agent")  # 不同的agent有不同的prompt
        # create_react_agent(); create_self_ask_with_search_agent();
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools,
                                       verbose=True, max_iterations=2)

        for chunk in agent_executor.stream(
                {"input": "what is the weather in SF and then LA"}):
            print(chunk)
            print("------")


from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.convert_to_openai import \
    format_tool_to_openai_function
from langchain.agents.format_scratchpad import \
    format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage


# 自定义工具(不能放到CustomAgent类中)
@tool
def get_word_length(word: str) -> int:
    """返回一个单词的长度."""
    return len(word)


class CustomAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(api_key=openai_api_key,
                              model="gpt-3.5-turbo-1106", temperature=0)

    def tool_as_openai_function(self):
        """将工具作为openai函数。"""
        from langchain.tools import MoveFileTool

        tools = [MoveFileTool()]
        functions = [format_tool_to_openai_function(t) for t in tools]

        message = self.llm.predict_messages(
            [HumanMessage(content="move file foo to bar")], functions=functions
        )

        return message

    def stateless_agent(self, query: str):
        tools = [get_word_length]  # 获得工具
        llm_with_tools = self.llm.bind(  # 将tool与大模型绑定
            functions=[format_tool_to_openai_function(t) for t in tools])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are very powerful assistant, but don't know current events"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = ({"input": lambda x: x["input"],
                  "agent_scratchpad": lambda
                      x: format_to_openai_function_messages(
                      x["intermediate_steps"]), }
                 | prompt
                 | llm_with_tools
                 | OpenAIFunctionsAgentOutputParser())  # 链式创建agent

        agent_executor = AgentExecutor(agent=agent, tools=tools,
                                       verbose=True, max_iterations=2)
        response = agent_executor.invoke({"input": query})

        return response

    def state_agent(self, query: str):
        tools = [get_word_length]  # 获得工具
        llm_with_tools = self.llm.bind(  # 将tool与大模型绑定
            functions=[format_tool_to_openai_function(t) for t in tools])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are very powerful assistant, but bad at calculating lengths of words.",),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), ])
        chat_history = []
        agent = ({"input": lambda x: x["input"],
                  "agent_scratchpad": lambda
                      x: format_to_openai_function_messages(
                      x["intermediate_steps"]),
                  "chat_history": lambda x: x["chat_history"], }
                 | prompt
                 | llm_with_tools
                 | OpenAIFunctionsAgentOutputParser())
        agent_executor = AgentExecutor(agent=agent, tools=tools,
                                       verbose=True, max_iterations=2)

        # 加入历史信息
        input1 = "how many letters in the word education?"
        result = agent_executor.invoke(
            {"input": input1, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=input1),
                             AIMessage(content=result["output"]), ])

        response = agent_executor.invoke(
            {"input": query, "chat_history": chat_history})

        return response


if __name__ == '__main__':
    api_key = ""
    text_dir = "xxx.pdf"
    query = "what is your name?"
    agent = Agent(api_key)
    print(agent.tool(text_dir, query))
    print(agent.stream_agent(text_dir, query))

    custom_agent = CustomAgent(api_key)
    # 将工具作为openai函数
    print(custom_agent.tool_as_openai_function())

    # CustomAgent-stateless_agent
    query = "How many letters in the word education?"
    print(custom_agent.stateless_agent(query))

    # CustomAgent-state_agent
    query = "is that a real word?"
    print(custom_agent.state_agent(query))
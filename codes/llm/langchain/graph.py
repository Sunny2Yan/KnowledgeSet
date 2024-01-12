# -*- coding: utf-8 -*-
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


class LangGraph:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key,
                              model="gpt-3.5-turbo-1106")

    def xxx(self):
        tools = [TavilySearchResults(max_results=1)]
        prompt = hub.pull("hwchase17/openai-functions-agent")

        # Construct agent
        agent_runnable = create_openai_functions_agent(self.llm, tools, prompt)

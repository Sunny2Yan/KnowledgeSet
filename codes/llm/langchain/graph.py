# -*- coding: utf-8 -*-
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]


prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Construct agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
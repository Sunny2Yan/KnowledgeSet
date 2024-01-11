# -*- coding: utf-8 -*-
"""SystemMessage, HumanMessage, AIMessage"""


class PromptTemplate:
    @staticmethod
    def string_prompt():
        """1. string prompt 支持 str.format"""
        from langchain.prompts import PromptTemplate

        string_template = PromptTemplate.from_template(
            "Tell me a {adjective} joke about {content}.")
        messages = string_template.format(
            adjective="funny", content="chickens")  # 没有变量直接 .format()
        return messages

    @staticmethod
    def chat_prompt_1():
        """2. chat prompt 定义 system, assistant, human (推荐)"""
        from langchain_core.prompts import ChatPromptTemplate

        chat_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"), ])

        messages = chat_template.format_messages(
            name="Bob", user_input="What is your name?")

        return messages

    @staticmethod
    def chat_prompt_2():
        """3. 除使用 (type, content) 外，还可以直接传入实例"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.prompts import HumanMessagePromptTemplate
        from langchain_core.messages import SystemMessage

        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat.")),
            HumanMessagePromptTemplate.from_template("{text}"), ])
        messages = chat_template.format_messages(
            text="I don't like eating tasty things")

        return messages

    @staticmethod
    def group_prompt():
        """4. 使用 pipeline 快速创建 (推荐)"""
        from langchain.schema import AIMessage, HumanMessage, SystemMessage

        system_prompt = SystemMessage(content="You are a nice pirate")
        human_prompt = HumanMessage(content="hi")
        assistant_prompt = AIMessage(content="what?")
        new_prompt = (system_prompt + human_prompt + assistant_prompt +
                      "{input}")
        messages = new_prompt.format_messages(input="i said hi")

        return messages


from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector


class FewShotTemplate:
    # 5-shot
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"}, ]

    def __init__(self):
        self.example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}", )

    def length_based_example_selector(self):
        """1. 按长度选择，较长的输入选择较少的示例，较短的输入选择更多的示例"""
        example_selector = LengthBasedExampleSelector(
            examples=self.examples,  # 可选择的示例
            example_prompt=self.example_prompt,  # 用于格式化示例的模板
            max_length=25,  # 格式化示例的最大长度
        )
        dynamic_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=self.example_prompt,
            prefix="Give the antonym of every input",
            suffix="Input: {adjective}\nOutput:",
            input_variables=["adjective"],
        )

        return dynamic_prompt.format(adjective='big')

    def xxx(self):
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            # The list of examples available to select from.
            examples,
            # The embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(),
            # The VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # The number of examples to produce.
            k=2,
        )
        mmr_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Give the antonym of every input",
            suffix="Input: {adjective}\nOutput:",
            input_variables=["adjective"],
        )


if __name__ == '__main__':
    prompt_template = PromptTemplate()
    print(prompt_template.string_prompt())
    print(prompt_template.chat_prompt_1())
    print(prompt_template.chat_prompt_2())
    print(prompt_template.group_prompt())

    # langchain 表达式支持 invoke() 调用
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "Tell me a {adjective} joke about {content}.")
    prompt_val = prompt.invoke(
        {"adjective": "funny", "content": "chickens"})
    print(prompt_val.to_messages())


    # few-shot learning
    few_shot_template = FewShotTemplate()
    print(few_shot_template.length_based_example_selector())

# -*- coding: utf-8 -*-
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
    few_shot_template = FewShotTemplate()
    print(few_shot_template.length_based_example_selector())
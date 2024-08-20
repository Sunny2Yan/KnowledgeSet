# -*- coding: utf-8 -*-
import re
from openai import OpenAI


class VLLMModel:
    def inference(self, messages: list[dict]):
        client = OpenAI(api_key="EMPTY", base_url="http://8000.lzx.aip.ennewi.cn/v1")
        models = client.models.list()
        model = models.data[0].id
        chat_response = client.chat.completions.create(
            messages=messages,
            model=model,
            n=1,
            temperature=1,
            stop=None,
            stream=False,
            )

        return chat_response


prompt = """
你在一个思考、行动、暂停、观察的循环中运行。
在循环结束时，你输出一个答案
使用思考来描述你对所问问题的想法。
使用行动来运行其中一个可用的操作 - 然后返回暂停。
观察将是运行这些操作的结果。

你可用的操作是：

计算：
例如计算：4 * 7 / 3
运行计算并返回数字 - 使用 Python，因此请确保在必要时使用浮点语法

平均狗体重：
例如平均狗体重：牧羊犬
在给定品种的情况下返回狗的平均体重

示例会话：

问题：斗牛犬的体重是多少？
想法：我应该使用 average_dog_weight 来查看狗的体重
动作：average_dog_weight：斗牛犬
暂停

您将再次收到以下信息：

观察：斗牛犬重 51 磅

然后您输出：

答案：斗牛犬重 51 磅
"""


class ReActAgent:
    def __init__(self, system=""):
        self.llm = VLLMModel()
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self) -> str:
        completion = self.llm.inference(self.messages)
        return completion.choices[0].message.content


def calculate(what):
    return eval(what)


def average_dog_weight(name):
    if name in "Scottish Terrier":
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "玩具贵宾犬":
        return("玩具贵宾犬的平均体重为 7 磅")
    else:
        return("An average dog weights 50 lbs")


known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

agent = ReActAgent(prompt)
result = agent("玩具贵宾犬有多重？")
print(result)
# 想法：我应该使用平均狗体重动作来查找玩具贵宾犬的平均体重。
# 动作：average_dog_weight：玩具贵宾犬
# 暂停
result = average_dog_weight("玩具贵宾犬")  # 手动执行

next_prompt = "Observation: {}".format(result)
agent(next_prompt)
# {'role': 'assistant', 'content': 'Answer: 玩具贵宾犬的平均体重为 7 磅'}

# -----------------以上就是一轮完整的react------------------------
# 自动调用
action_re = re.compile('^Action: (\w+): (.*)$')   # re selection action

def query(question, max_rounds=5):
    round = 0
    bot = ReActAgent(prompt)
    next_prompt = question
    while round < max_rounds:
        round += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


question = """我有两只狗，一只边境牧羊犬和一只苏格兰梗犬。它们的总体重是多少"""
query(question)

# 想法：我需要找到边境牧羊犬和苏格兰梗的平均体重，然后将它们加在一起得到总体重。
# 动作：average_dog_weight：边境牧羊犬
# 暂停
# -- 运行 average_dog_weight 边境牧羊犬
# 观察：边境牧羊犬的平均体重为 37 磅
# 想法：现在我需要找到苏格兰梗的平均体重。
# 动作：average_dog_weight：苏格兰梗
# 暂停
# -- 运行 average_dog_weight 苏格兰梗
# 观察：苏格兰梗平均体重 20 磅
# 想法：我现在知道了两只狗的平均体重。我将把它们加在一起得到总体重。
# 动作：计算：37 + 20
# 暂停
# -- 运行计算 37 + 20
# 观察：57
# 答案：边境牧羊犬和苏格兰梗的总体重为 57 磅。
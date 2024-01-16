# -*- coding: utf-8 -*-

import chainlit as cl


# 1. @step用于assistant采取一系列步骤来处理user的请求，具有输入输出。
@cl.step
async def tool():
    await cl.sleep(2)

    return "Response from the tool!"


# 2. @session用于防止多个用户同时与机器人聊天，信息糅杂。
@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("counter", 0)  # 设置session中公共元素


@cl.on_message
async def on_message(message: cl.Message):
    tool_res = await tool()  # 调用step中的tool

    counter = cl.user_session.get("counter")  # 在session中获得公共元素
    counter += 1
    cl.user_session.set("counter", counter)

    await cl.Message(content=f"You sent {counter} message(s)!").send()


# 3. action用于创建按钮，user点击时触发
@cl.on_chat_start
async def start():
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="action_button", value="example_value", description="Click me!")
    ]

    await cl.Message(content="Interact with this action button:", actions=actions).send()


@cl.action_callback("action_button")
async def on_action(action: cl.Action):
    print("The user clicked on the action button!")

    return "Thank you for clicking on the action button!"


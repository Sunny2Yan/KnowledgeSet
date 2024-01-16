# -*- coding: utf-8 -*-

import chainlit as cl
from chainlit.types import ThreadDict


@cl.on_chat_start
def on_chat_start():
    """在创建新聊天会话时调用的钩子，以显示元素为例。"""
    # display=side  元素名显示为链接，点击时出现在消息一侧；
    # display=page  元素名显示为链接，点击时重定向到新的页面；
    # display=inline  元素显示在消息中
    image = cl.Image(path="./cat.jpeg", name="image1", display="inline")

    # 将图像附加到消息中（text、image、pdf...）
    await cl.Message(
        content="This message has an image!",
        elements=[image], ).send()


@cl.on_message
def on_message(message: cl.Message):
    """用户接收新消息时调用的钩子"""
    # 1.简单的回复用户消息
    response = f"Hello, you just sent: {message.content}!"
    await cl.Message(response).send()  # assistant发送给user的消息

    # 2. 在等待响应时显示加载程序
    msg = cl.Message(content="")
    await msg.send()

    await cl.sleep(2)  # do some work

    msg.content = f"Processed message {message.content}"

    await msg.update()




@cl.on_stop
def on_stop():
    """定义用户在任务运行时单击停止按钮时调用的钩子"""
    ...


@cl.on_chat_end
def on_chat_end():
    """聊天会话结束时，用户断开连接或启动了新的聊天会话调用的钩子"""
    ...


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """用户恢复以前断开连接的聊天会话调用的钩子。仅当启用身份验证和数据持久性时，才可用"""
    ...

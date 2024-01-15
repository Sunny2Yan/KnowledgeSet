import gradio as gr


"""简单调用高级接口（不好用！）"""


def greet(name):
    return "Hello " + name + "!"


# 1. 含一个输入框一个输出框
demo_1 = gr.Interface(fn=greet, inputs="text", outputs="text")

# 2. 记录输入输出到 ./flagged/log.csv 文件中 SampleCSVLogger为简易版
demo_2 = gr.Interface(fn=greet, inputs="text", outputs="text",
                       flagging_callback=gr.CSVLogger())


tts_examples = ["I love learning machine learning",
                "How do you do?",]

tts_demo = gr.load(  # 从 huggingface 拉取模型
    name="huggingface/facebook/fastspeech2-en-ljspeech",  # 从huggingface下载模型
    title=None,
    examples=tts_examples,
    description="Give me something to say!",
)

stt_demo = gr.load(
    name="huggingface/facebook/wav2vec2-base-960h",
    title=None,
    inputs="mic",
    description="Let me try to guess what you're saying!",
)

# 3. 展示不同模型供选择
demo_3 = gr.TabbedInterface([tts_demo, stt_demo], ["Text-to-speech", "Speech-to-text"])

# 4. 一个输入 name，两个不同的输出
greeter_1 = gr.Interface(lambda name: f"Hello {name}!", inputs="textbox", outputs=gr.Textbox(label="Greeter 1"))
greeter_2 = gr.Interface(lambda name: f"Greetings {name}!", inputs="textbox", outputs=gr.Textbox(label="Greeter 2"))
demo_4 = gr.Parallel(greeter_1, greeter_2)  # 上下平行输出

# 5. 多个interface串行，只输出最后一个
get_name = gr.Interface(lambda name: name, inputs="textbox", outputs="textbox")
prepend_hello = gr.Interface(lambda name: f"Hello {name}!", inputs="textbox", outputs="textbox")
append_nice = gr.Interface(lambda greeting: f"{greeting} Nice to meet you!",
                           inputs="textbox", outputs=gr.Textbox(label="Greeting"))
demo_5 = gr.Series(get_name, prepend_hello, append_nice)


if __name__ == '__main__':
    demo_5.launch(server_name='0.0.0.0', server_port=8000)
    # launch(
    # share=False  # 创建可公开共享的链接（SSH 隧道）
    # max_threads  # 并行生成的最大总线程数（默认 40）
    # show_error   # 界面中错误会以警报模式输出在浏览器控制台日志中（默认 False）
    # )

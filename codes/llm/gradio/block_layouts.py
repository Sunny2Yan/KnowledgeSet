import gradio as gr


"""Blocks是Gradio的低级API，允许创建自定义Web应用程序和演示（经常使用）"""


def update(name):
    return f"Welcome to Gradio, {name}!"


with gr.Blocks(title='nihao') as demo_0:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")  # 按钮
    btn.click(fn=update, inputs=inp, outputs=out)  # 按钮触发函数


# 1.1 with 创建 Blocks() 块，所有组件放到块下面（组件在后续介绍）
# gr.Blocks(
#     theme,
#     mode,
#     title,
#     css='style.css'  # 页面风格可以自定义
# )


"""Blocks() 块下面的页面布局"""


# 2.1 Row()实现行布局
with gr.Blocks() as demo_1:
    with gr.Row():
        gr.Textbox()  # 两个文本框位于同一行
        gr.Textbox()

gr.Row(
    variant='default',  # 'default': 无背景, 'panel': 灰色圆角, 'compact': 圆角紧凑
    visible=True,       # 是否可见, default=True
    equal_height=True   # 子元素是否具有相同的高度，default=True
)


# 2.1 Column()实现列布局
with gr.Blocks() as demo_2:
    with gr.Row():  # 同一行下有两个列组
        with gr.Column(scale=1):
            text1 = gr.Textbox()
            text2 = gr.Textbox()
        with gr.Column(scale=4):
            btn1 = gr.Button("Button 1")
            btn2 = gr.Button("Button 2")

gr.Column(
    scale=1,  # 与相邻列相比的相对宽度，1表示等宽
    min_width=320,  # 列的最小像素宽度，不够则换行
    variant='default',  # 'default': 无背景, 'panel': 灰色圆角, 'compact': 圆角紧凑
    visible=True,       # 是否可见, default=True
)


# 2.3 Group()创建一个组，使子项组合，之间没有任何填充或边距
with gr.Blocks() as demo_3:
    with gr.Group():
        gr.Textbox(label="First")
        gr.Textbox(label="Last")

gr.Group(
    visible=True  # 组是否可见，default=True, False则被隐藏
)


# 2.4 Tab()创建一个标签页，左右折叠多个标签页，可供选择
with gr.Blocks() as demo_4:
    with gr.Tab(label="页面一"):
        gr.Textbox()
        gr.Button("summit")
    with gr.Tab("页面二"):
        gr.Textbox()
        gr.Button("summit")

gr.Tab(
    label="",  # 选项卡的标签名
    id=None  # 选项卡的标识符, 默认无
)


# 2.5 Box()创建一个带圆角的盒子，将子元素框住
with gr.Blocks() as demo_5:
    with gr.Box():  # 在一个框里含两个文本框
        gr.Textbox(label="First")
        gr.Textbox(label="Last")

gr.Box(
    vasiable=True  # 是否可见，default=True
)


# 2.6 Accordion() 可以切换以显示/隐藏包含的内容
with gr.Blocks() as demo_6:
    with gr.Accordion("See Details"):  # 一个 ”See Details“的倒三角，点击则展开
        gr.Markdown("lorem ipsum")

gr.Accordion(
    label='xxx',  # 显示的标签
    open=True,    # 默认折叠是否处于打开状态，default=True
    visible=True  # 是否可见，default=True
)


if __name__ == "__main__":
    demo_6.launch(server_name='0.0.0.0', server_port=8000)
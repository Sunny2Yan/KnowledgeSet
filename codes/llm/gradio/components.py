import gradio as gr


"""gradio的核心模块，与函数交互"""


# 1. 文本输入类组件
# 1.1 Markdown() 输出markdown格式文本
gr.Markdown(
    value='',  # markdown 文本
    visible=True,
    rtl=False,  # 从右向左排列
)
# 1.2 Number() 创建数值字段，供用户输入数值
gr.Number(
    value=None,  # 初始数值
    label=None,  # 组件名称
    info=None,   # 其他说明
    show_label=False,  # 是否展示标签
    container=True,  # 将组件放到容器中，边缘有填充
    scale=None,  # 与行中相邻组件的相对宽度比
    interactive=None,  # True为可编辑，False为不可编辑
    visible=True,

)
# 1.3 Textbox() 文本框，输入字符串
# 1.4 HighlightedText() 对文本进行高亮处理
# 1.5 Chatbot() 创建多轮对话的聊天框
# 1.6 Code() 创建用于输入、编辑或查看的代码编辑器


# 2. 选择类组件
# 2.1 Radio() 单选框，多个互斥
# 2.2 Checkbox() 创建复选框（左边口，可点击√）
# 2.3 CheckboxGroup() 创建复选框组, 包含多个判断项（可用gr.Group()实现）
# 2.4 Slider() 创建滑块，左右滑动
# 2.5 Dropdown() 创建下拉列表，可供选择
# 2.6 ColorPicker() 创建颜色选择器，点击可在色彩盘选择颜色


# 3. 按钮类组件
# 3.1 Button() 创建可供点击的按钮
# 3.2 ClearButton() 创建清除按钮（同Button）
# 3.3 DuplicateButton() 创建复制按钮，
# 3.4 UploadButton() 上传按钮


# 4. 上传类组件
# 4.1 File() 上传文件组件
# 4.2 Audio() 上传音频组件
# 4.3 Image() 上传图像组件
# 4.4 Video() 上传视频组件


# 5. 展示类组件
# 5.1 AnnotatedImage() 带有注释的图像（一张图）
# 5.2 Gallery() 展示图片，用于浏览图库（多图）
# 5.3 make_waveform() 将音频文件生成视频波形
# 5.4 Dataset() 创建数据集展示
# 5.5 JSON() 输出json格式的组件


# 6. 作表、作图类组件
# 6.1 Dataframe() 创建表格
# 6.2 Label() 输出标签及置信度
# 6.3 BarPlot() 创建条形图
# 6.4 ScatterPlot() 创建散点图
# 6.5 LinePlot() 创建线图
# 6.6 Plot() 绘制matplotlib格式的图


# 7. 函数中日志类组件
# 7.1 Info() 输出要显示的信息，用于函数
# 7.2 Warning() 输出警告信息，用于函数
# 7.3 Error() 输出错误信息，用于函数中


# 8. 其他类组件
# 8.1 Examples() 创建示例
# 8.2 State() 隐藏变量组件，用户刷新页面时，将清除 State 变量的值
# 8.3 Progress() 进度跟踪器，类似与tqdm
# 8.4 load() 从huggingface拉取模型
# 8.5 update() 更新输出组件的值，注意是组件！


# For Example
def parser_pdf(files):
    outputs = {}
    for file in files:
        root = file.name
        file_name = file.name.split('/')[-1]
        postfix = file.name.split('.')[-1]
        if postfix.lower() == 'pdf':
            text_all = ''
            with pdfplumber.open(root) as pdf_reader:
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    text_all += text

            outputs[file_name] = text_all.strip()

        elif postfix.lower() == 'doc' or postfix.lower() == 'docx':
            if postfix.lower() == 'doc':
                from win32com import client as wc
                word = wc.Dispatch('Word.Application')
                doc = word.Documents.Open(root)
                doc.SaveAs(root + 'x', 12, False, "", True, "", False,
                           False, False, False)
                doc.Close()
                word.Quit()
                file_path = root + 'x'
            doc_reader = docx.Document(file_path)
            text_all = ''
            for para in doc_reader.paragraphs:
                text_all += para.text
            outputs[file_name] = text_all.strip()
    return outputs


def text_connect(file_text: dict, string: str):
    all_text = "下面文本含有重要信息，请根据下面文本使用中文回答相关问题。\n"
    for text_name, text in file_text.items():
        all_text += text

    return all_text + '\n' + string


def fnc(x):
    return x


with gr.Blocks() as demo:
    with gr.Row():
        input_file = gr.File(file_count="multiple",
                             file_types=[".pdf", ".doc", ".docx"],
                             scale=5)
        input_text = gr.Textbox(
            placeholder="请输入你的问题",
            lines=10,
            container=False,
            show_label=False,
            scale=5,
        )

    with gr.Row():
        submit_button = gr.Button("Submit", variant="primary", scale=1,
                                  min_width=0)
    with gr.Row():
        output_text = gr.Textbox(
            lines=5,
            container=False,
            show_label=False,
            scale=10,
        )

    saved_input = gr.State()

    button_event_preprocess = (
        submit_button.click(
            fn=parser_pdf,
            inputs=input_file,
            outputs=saved_input,
            api_name=False,
            queue=False,
        )
        .then(
            fn=text_connect,
            inputs=[saved_input, input_text],
            outputs=saved_input,
            api_name=False,
            queue=False,
        )
        .then(
            fn=fnc,
            inputs=saved_input,
            outputs=output_text,
            api_name=False,
            queue=False,
        )
    )

demo.launch(server_name='0.0.0.0', server_port=8000)


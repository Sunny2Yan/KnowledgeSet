import gradio as gr


"""Base()创建主题样式"""



def create_tag(label):
    tag = gr.Tab(label=label, visible=True)
    with tag:
        gr.Textbox(value="111")
    return tag


with gr.Blocks() as demo:
    tag1, parameter_list = create_tag("1")
    tag2 = create_tag("2")
    tag3 = create_tag("3")

    dropdown = gr.Dropdown(choices=[0, 1, 2], value=0)

    def change(index):
        result = [gr.update(visible=False) for _ in range(3)]
        result[index] = gr.update(visible=True)
        return result
    dropdown.change(change, dropdown, [tag1, tag2, tag3])


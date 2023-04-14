import gradio as gr
from transformers import AutoModel, AutoTokenizer

from utils.toolbox import format_io

tokenizer_glm = AutoTokenizer.from_pretrained("checkpoints/chatglm-6b", trust_remote_code=True)
model_glm = AutoModel.from_pretrained("checkpoints/chatglm-6b", trust_remote_code=True).half().cuda()
model_glm = model_glm.eval()


# Define function to generate model predictions and update the history
def predict_glm_stream(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []
    history = list(map(tuple, history))
    for response, updates in model_glm.stream_chat(
            tokenizer_glm, input, history, top_p=top_p, temperature=temperature, max_length=max_length):
        yield updates


def reset_textbox():
    return gr.update(value="")


title = """<h1 align="center"> 🚀CHatGLM-6B - A Streaming Chatbot with Gradio </h1>
<h2 align="center">Enhance User Experience with Streaming and customizable Gradio Themes</h2>"""
header = """<center>Find more about Chatglm-6b on Huggingface at <a href="https://huggingface.co/THUDM/chatglm-6b" target="_blank">THUDM/chatglm-6b</a>, and <a href="https://github.com/THUDM/ChatGLM-6B" target="_blank">here</a> on Github.<center>"""
description = """<br>
ChatGLM-6B is an open-source, Chinese-English bilingual dialogue language model based on the General Language Model (GLM) architecture with 6.2 billion parameters. 
However, due to the small size of ChatGLM-6B, it is currently known to have considerable limitations, such as factual/mathematical logic errors, possible generation of harmful/biased content, weak contextual ability, self-awareness confusion, and Generate content that completely contradicts Chinese instructions for English instructions. Please understand these issues before use to avoid misunderstandings. A larger ChatGLM based on the 130 billion parameter GLM-130B is under development in internal testing.
"""

theme = gr.themes.Default(  # color contructors
    primary_hue="violet",
    secondary_hue="indigo",
    font=["ui-sans-serif", "system-ui", "sans-serif", gr.themes.utils.fonts.GoogleFont("Source Sans Pro")],
    font_mono=["ui-monospace", "Consolas", "monospace", gr.themes.utils.fonts.GoogleFont("IBM Plex Mono")],
    neutral_hue="purple").set(slider_color="#800080")

gr.Chatbot.postprocess = format_io

with gr.Blocks(css="""#col_container {margin-left: auto; margin-right: auto;}
                #chatglm {height: 520px; overflow: auto;} """, theme=theme) as demo:
    gr.HTML(title)
    gr.HTML(header)
    with gr.Column():  # (scale=10):
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=8):
                    inputs = gr.Textbox(placeholder="Hi there!", label="Type an input and press Enter ⤵️ ")
                with gr.Column(scale=1):
                    b1 = gr.Button('🏃Run', elem_id='run').style(full_width=True)
                with gr.Column(scale=1):
                    b2 = gr.Button('🔄Clear the Chatbot!', elem_id='clear').style(full_width=True)
                    state_glm = gr.State([])

        with gr.Box():
            chatbot_glm = gr.Chatbot(elem_id="chatglm", label='THUDM-ChatGLM6B')

        with gr.Accordion(label="Parameters for ChatGLM-6B", open=False):
            max_length = gr.Slider(
                0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True,  visible=True
            )
            top_p = gr.Slider(
                0, 1, value=0.7, step=0.01, label="Top P", interactive=True, visible=True
            )
            temperature = gr.Slider(
                0, 1, value=0.95, step=0.01, label="Temperature", interactive=True,  visible=True
            )

    inputs.submit(predict_glm_stream,
                  [inputs, max_length, top_p, temperature, chatbot_glm],
                  [chatbot_glm], )
    inputs.submit(reset_textbox, [], [inputs])

    b1.click(predict_glm_stream,
             [inputs, max_length, top_p, temperature, chatbot_glm],
             [chatbot_glm], )
    b1.click(reset_textbox, [], [inputs])

    b2.click(lambda: None, None, chatbot_glm, queue=False)

    gr.Markdown(description)

if __name__ == "__main__":
    demo.queue(concurrency_count=16).launch(height=800, debug=True, server_name="0.0.0.0")

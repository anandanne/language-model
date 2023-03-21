import gradio as gr
from transformers import GPT2LMHeadModel, BertTokenizerFast

from basaran.choice import reduce_choice
from basaran.model import StreamModel

model_name_or_path = "../../models/gpt2-abstract/checkpoint-500000"
tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
stream_model = StreamModel(model, tokenizer, remove_whitespace=True)


def user(user_message, history):
    user_message = "【提示】：" + user_message
    return "", history + [[user_message, None]]


def generate(min_length, max_length, top_p, temperature, history):
    prompt = history[-1][0][5:]
    buffers = {}
    for choice in stream_model(
        prompt,
        min_tokens=min_length,
        max_tokens=max_length,
        top_p=top_p,
        temperature=temperature,
    ):
        index = choice["index"]
        if index not in buffers:
            buffers[index] = []
        buffers[index].append(choice)

        response = reduce_choice(buffers[index])["text"]
        history[-1][1] = "【续写】：" + prompt + response

        yield history


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center"> GPT2 Based Abstract Generation </h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="GPT2论文摘要续写")
            msg = gr.Textbox(placeholder="论文摘要开头", label="提示")
            clear = gr.Button("清除历史记录")

        with gr.Column(scale=1):
            min_length = gr.Slider(0, 256, value=1, step=1.0, label="Minimum length", interactive=True)
            max_length = gr.Slider(0, 512, value=256, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        generate, [min_length, max_length, top_p, temperature, chatbot], chatbot
    )
    clear.click(lambda: None, None, chatbot)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")

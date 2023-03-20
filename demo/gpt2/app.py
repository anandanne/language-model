import gradio as gr
from transformers import GPT2LMHeadModel, BertTokenizerFast
from demo.stream import StreamModelForCausalLM

model_name_or_path = "../../models/gpt2-abstract/checkpoint-358000"
tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
stream_model = StreamModelForCausalLM(model, tokenizer)


def predict(max_length, top_p, temperature, history):
    prompt = history[-1][0]
    for response in stream_model.stream_decode(
        prompt,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ):
        yield response


def infer_model(input_text, max_length=128, top_p=0.9):
    inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=False).to(model.device)
    gen = model.generate(
        inputs=inputs['input_ids'],
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
        early_stopping=True)[0].tolist()

    text = tokenizer.decode(gen[len(inputs['input_ids'][0]):])

    return input_text + text.replace(" ", "")


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center"> GPT2 Based Abstract Generation </h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="GPT2论文摘要续写")
            msg = gr.Textbox(placeholder="中国道家文化作为中国诸多文化体系中的主体之一")
            clear = gr.Button("清除历史记录")

        with gr.Column(scale=1):
            max_length = gr.Slider(0, 512, value=256, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    def user(user_message, history):
        user_message = "【提示】：" + user_message
        return "", history + [[user_message, None]]

    def bot(max_length, top_p, temperature, history):
        prompt = history[-1][0][5:]
        for response in stream_model.stream_decode(
            prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ):
            history[-1][1] = "【续写】：" + prompt + response.replace(" ", "")
            yield history


    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        bot, [max_length, top_p, temperature, chatbot], chatbot
    )
    clear.click(lambda: None, None, chatbot)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")

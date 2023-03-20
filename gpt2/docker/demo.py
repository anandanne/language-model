import gradio as gr
from transformers import GPT2LMHeadModel, BertTokenizerFast

model_name_or_path = "outputs/gpt2-chinese"
tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)


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
    chatbot = gr.Chatbot(label="GPT2论文摘要续写")
    msg = gr.Textbox(placeholder="中国道家文化作为中国诸多文化体系中的主体之一")
    clear = gr.Button("清除历史记录")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = infer_model(history[-1][0])
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

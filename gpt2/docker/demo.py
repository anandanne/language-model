import gradio as gr
from transformers import GPT2LMHeadModel

from tokenization import CpmTokenizer

model_name_or_path = "outputs/gpt2-chinese"
tokenizer = CpmTokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)


def infer_model(input_text, max_length=128, top_p=0.9):
    inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=False).to(model.device)
    gen = model.generate(
        inputs=inputs['input_ids'],
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=0,
        no_repeat_ngram_size=2,
        early_stopping=True)[0]

    return input_text + tokenizer.decode(gen[len(inputs['input_ids'][0]):])


demo = gr.Interface(
    infer_model,
    [gr.Textbox(placeholder="Enter sentence here...", lines=5)],
    ["text"],
    examples=[
        ["祁连山是我国著名的多金属成矿带之一，"],
        ["本学期,我们采用班级授课的形式展开了IrobotQ3D虚拟仿真机器人的课堂教学实践。"],
        ["中国道家文化作为中国诸多文化体系中的主体之一,"],
    ],
    title="GPT2 Based Abstract Generation",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

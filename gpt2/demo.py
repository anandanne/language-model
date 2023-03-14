import gradio as gr
from transformers import GPT2LMHeadModel, AutoTokenizer

model_name_or_path = "outputs/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)


def infer_model(input_text, max_length=128, top_p=0.9):
    inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=False).to(model.device)
    gen = model.generate(
        inputs=inputs['input_ids'],
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        eos_token_id=102,
        pad_token_id=0,
        no_repeat_ngram_size=2,
        early_stopping=True)[0]

    gen = gen[len(inputs['input_ids'][0]):]
    res = []
    for g in gen:
        if g.item() == 102:
            break
        else:
            res.append(g)

    sentence = tokenizer.batch_decode([res])[0]

    return input_text + sentence.replace(" ", "")


demo = gr.Interface(
    infer_model,
    [gr.Textbox(placeholder="Enter sentence here...", lines=5)],
    ["text"],
    examples=[
        ["【燕姞贻天梦，梁王尽孝思。"],
    ],
    title="GPT2 Based Generation",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

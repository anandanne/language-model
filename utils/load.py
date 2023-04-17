import torch
from peft import PeftModel
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)


def load_llama_tokenizer_and_model(base_model, adapter_model=None, load_8bit=False, device="cuda:0"):

    if adapter_model:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(adapter_model)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if adapter_model:
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

        if model_vocab_size != tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            model.resize_token_embeddings(tokenzier_vocab_size)

        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            torch_dtype=torch.float16,
        )

    if device == "cpu":
        model.float()

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.to(device)
    model.eval()

    return tokenizer, model

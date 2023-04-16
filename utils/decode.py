import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    generated_tokens = []
    past_key_values = None
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)

        yield text
        if any([x in text for x in stop_words]):
            return


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False


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

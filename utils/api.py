import json
import secrets
import time
from typing import List, Union

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizer


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    prompt: Union[str, List[str]] = None
    stream: bool = None
    max_tokens: int = 2048
    temperature: float = 0.95
    top_p: float = 0.7


class ChatBody(BaseModel):
    messages: List[Message] = None
    stream: bool = None
    max_tokens: int = 2048
    temperature: float = 0.95
    top_p: float = 0.7


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def get_completion_template(model_name, chat=False):
    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "text_completion",
        "created": round(time.time()),
        "model": model_name,
        "choices": [],
    }
    if chat:
        template["object"] = "chat.completion"

    return template


def serialize(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def map_choice(text, delta=None, chat=False):
    """Create a choice object from model outputs."""
    choice = {
        "index": 0,
        "finish_reason": None,
    }

    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
        choice["logprobs"] = None

    if delta is not None:
        choice["delta"] = {"role": "assistant", "content": delta}

    return choice


def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_cache: bool = None,
):
    generated_tokens = []
    past_key_values = None
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids, use_cache=use_cache)
            else:
                outputs = model(
                    input_ids[:, -1:], past_key_values=past_key_values, use_cache=use_cache,
                )

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

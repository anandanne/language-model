import json
import logging
import secrets
import time
import warnings
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from utils.decode import sample_decode, load_tokenizer_and_model

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer, model, device = load_tokenizer_and_model(
    "checkpoints/llama-7b-hf", "checkpoints/lora/chinese-alpaca-lora-7b"
)
system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
user_prompt = "### Instruction:\n\n{}\n\n### Response:\n\n"
assistant_prompt = "{}\n\n"
stop_words = ["### Instruction", "### Response", "</s><s>"]
end_of_text = "</s><s>"


def serialize(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def map_choice(text):
    """Create a choice object from model outputs."""
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": None,
    }

    return choice


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    messages: List[Message] = None
    stream: bool = None
    max_tokens: int = 1000
    temperature: float = 0.2
    top_p: float = 0.9


@app.post("/v1/chat/completions")
async def completions(body: Body, request: Request):
    question = body.messages[-1]
    question = question.content
    max_tokens = max(1000, body.max_tokens)

    context = system_prompt
    for x in body.messages:
        if x.role == "user":
            context += user_prompt.format(x.content)
        else:
            context += assistant_prompt.format(x.content)

    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": round(time.time()),
        "model": "chinese-vicuna",
        "choices": [],
    }
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    async def event_generator():
        for response in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=stop_words,
            max_length=max_tokens,
            top_p=getattr(body, "top_p", 0.2),
            temperature=getattr(body, "temperature", 0.9),
        ):
            data = template.copy()
            response = response.replace(end_of_text, "")
            data["choices"] = [map_choice(response)]

            if await request.is_disconnected():
                return
            yield serialize(data)

        yield ""

    if body.stream:
        return EventSourceResponse(event_generator())
    else:
        response = ""
        for r in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=stop_words,
            max_length=max_tokens,
            top_p=getattr(body, "top_p", 0.7),
            temperature=getattr(body, "temperature", 0.95),
        ):
            response = r

        data = template.copy()
        response = response.replace(end_of_text, "")
        data["choices"] = [map_choice(response)]

        data["usage"] = {
            "prompt_tokens": len(question),
            "completion_tokens": len(response),
            "total_tokens": len(question) + len(response),
        }

        return data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)

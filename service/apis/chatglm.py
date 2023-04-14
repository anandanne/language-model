import json
import secrets
import time
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModel, AutoTokenizer

model_path = "checkpoints/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    messages: List[Message] = None
    stream: bool = None
    max_tokens: int = 2048
    temperature: float = 0.95
    top_p: float = 0.7


def serialize(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def map_choice(text, delta=None):
    """Create a choice object from model outputs."""
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": None,
    }

    if delta is not None:
        choice["delta"] = {"role": "assistant", "content": delta}

    return choice


@app.post("/v1/chat/completions")
async def completions(body: Body, request: Request):
    question = body.messages[-1]
    question = question.content

    user_question = ''
    history = []
    for message in body.messages:
        if message.role == 'user':
            user_question = message.content
        elif message.role == 'system' or message.role == 'assistant':
            assistant_answer = message.content
            history.append((user_question, assistant_answer))

    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": round(time.time()),
        "model": model_path,
        "choices": [],
    }

    async def event_generator():
        size = 0
        for response, _ in model.stream_chat(
            tokenizer,
            question,
            history,
            max_length=max(2048, body.max_tokens),
            top_p=getattr(body, "top_p", 0.7),
            temperature=getattr(body, "temperature", 0.95),
        ):
            delta = response[size:]
            size = len(response)
            data = template.copy()
            data["choices"] = [map_choice(response, delta)]

            if await request.is_disconnected():
                return
            yield serialize(data)

        yield "[DONE]"

    if body.stream:
        return EventSourceResponse(event_generator())
    else:
        response, _ = model.chat(
            tokenizer,
            question,
            history,
            max_length=max(2048, body.max_tokens),
            top_p=getattr(body, "top_p", 0.7),
            temperature=getattr(body, "temperature", 0.95),
        )
        data = template.copy()
        data["choices"] = [map_choice(response)]

        data["usage"] = {
            "prompt_tokens": len(question),
            "completion_tokens": len(response),
            "total_tokens": len(question) + len(response),
        }

        return data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)

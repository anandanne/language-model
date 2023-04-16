import json
from typing import List, Union

from pydantic import BaseModel


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

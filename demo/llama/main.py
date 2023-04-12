import json
import logging
import secrets
import time
import warnings
from dataclasses import dataclass, field
from typing import List
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import HfArgumentParser

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={
            "help": "prompt structure given user's input text"
        },
    )
    end_string: Optional[str] = field(
        default="\n\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )


pipeline_name = "inferencer"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((
    ModelArguments,
    PipelineArguments,
    ChatbotArguments,
))
model_args, pipeline_args, chatbot_args = (
    parser.parse_args_into_dataclasses()
)

with open(pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(
    model_args,
    tune_strategy='none',
    ds_config=ds_config,
    device=pipeline_args.device,
)

# We don't need input data, we will read interactively from stdin
data_args = DatasetArguments(dataset_path=None)
dataset = Dataset(data_args)

inferencer = AutoPipeline.get_pipeline(
    pipeline_name=pipeline_name,
    model_args=model_args,
    data_args=data_args,
    pipeline_args=pipeline_args,
)

# Chats
model_name = model_args.model_name_or_path
if model_args.lora_model_path is not None:
    model_name += f" + {model_args.lora_model_path}"

guide_message = (
    "\n"
    f"#############################################################################\n"
    f"##   A {model_name} chatbot is now chatting with you!\n"
    f"#############################################################################\n"
    "\n"
)
print(guide_message, end="")

# context = (
#     "You are a helpful assistant who follows the given instructions"
#     " unconditionally."
# )


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


def map_choice(text):
    """Create a choice object from model outputs."""
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": None,
    }

    return choice


@app.post("/v1/chat/completions")
async def completions(body: Body, request: Request):
    question = body.messages[-1]
    question = question.content
    max_tokens = max(1000, body.max_tokens)

    history = []
    for x in body.messages:
        if x["role"] == "user":
            history.append(f"Input: {x['content']}")
        else:
            history.append(f"Output: {x['content']}")

    context = "\n\n".join(history)
    context += "\n\nOutput: "

    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": round(time.time()),
        "model": model_args.model_name_or_path,
        "choices": [],
    }

    async def event_generator():
        partial_text = ""
        step = 1
        for _ in range(0, max_tokens, step):
            input_dataset = dataset.from_dict(
                {"type": "text_only", "instances": [{"text": context + partial_text}]}
            )
            output_dataset = inferencer.inference(
                model=model,
                dataset=input_dataset,
                max_new_tokens=max_tokens,
                top_p=getattr(body, "top_p", 0.7),
                temperature=getattr(body, "temperature", 0.95),
            )
            response = output_dataset.to_dict()["instances"][0]["text"]
            if response == "" or response == chatbot_args.end_string:
                break
            partial_text += response

            data = template.copy()
            data["choices"] = [map_choice(partial_text)]

            if await request.is_disconnected():
                return
            yield serialize(data)

        yield ""

    if body.stream:
        return EventSourceResponse(event_generator())
    else:
        input_dataset = dataset.from_dict(
            {"type": "text_only", "instances": [{"text": context}]}
        )

        output_dataset = inferencer.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=max_tokens,
            top_p=getattr(body, "top_p", 0.7),
            temperature=getattr(body, "temperature", 0.95),
        )

        response = output_dataset.to_dict()["instances"][0]["text"]

        data = template.copy()
        data["choices"] = [map_choice(response)]

        data["usage"] = {
            "prompt_tokens": len(question),
            "completion_tokens": len(response),
            "total_tokens": len(question) + len(response),
        }

        return data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, app_dir=".")

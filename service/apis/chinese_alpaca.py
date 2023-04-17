import argparse
import logging
import secrets
import time
import warnings

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from utils.api import Body, ChatBody, serialize, map_choice, sample_decode
from utils.load import load_llama_tokenizer_and_model

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/completions")
async def completions(body: Body, request: Request):
    prompt = body.prompt
    if isinstance(prompt, list):
        prompt = prompt[0]
    max_tokens = max(1000, body.max_tokens)

    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": round(time.time()),
        "model": f"{args.model_path}-{args.lora_model_path}",
        "choices": [],
    }

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(args.device)

    async def event_generator():
        size = 0
        for response in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=stop_words,
            max_length=max_tokens,
            top_p=getattr(body, "top_p", 0.2),
            temperature=getattr(body, "temperature", 0.9),
        ):
            response = response.replace(end_of_text, "")
            delta = response[size:]
            size = len(response)
            data = template.copy()
            data["choices"] = [map_choice(delta)]

            if await request.is_disconnected():
                return
            yield serialize(data)

        torch_gc()
        yield "[DONE]"

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
            top_p=getattr(body, "top_p", 0.2),
            temperature=getattr(body, "temperature", 0.9),
        ):
            response = r

        data = template.copy()
        response = response.replace(end_of_text, "")
        data["choices"] = [map_choice(response)]

        data["usage"] = {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(response),
            "total_tokens": len(prompt) + len(response),
        }

        torch_gc()

        return data


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody, request: Request):
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
        "model": f"{args.model_path}-{args.lora_model_path}",
        "choices": [],
    }

    input_ids = tokenizer(context, return_tensors="pt").input_ids
    input_ids = input_ids.to(args.device)

    async def chat_event_generator():
        size = 0
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
            delta = response[size:]
            size = len(response)
            data["choices"] = [map_choice(response, delta, chat=True)]

            if await request.is_disconnected():
                return
            yield serialize(data)

        torch_gc()
        yield "[DONE]"

    if body.stream:
        return EventSourceResponse(chat_event_generator())
    else:
        response = ""
        for r in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=stop_words,
            max_length=max_tokens,
            top_p=getattr(body, "top_p", 0.2),
            temperature=getattr(body, "temperature", 0.9),
        ):
            response = r

        data = template.copy()
        response = response.replace(end_of_text, "")
        data["choices"] = [map_choice(response, chat=True)]

        data["usage"] = {
            "prompt_tokens": len(context),
            "completion_tokens": len(response),
            "total_tokens": len(context) + len(response),
        }

        torch_gc()

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple API server for LLaMA')
    parser.add_argument('--model_path', '-m', type=str, help='基础模型文件所在路径',
                        default='/workspace/checkpoints/llama-7b-hf')
    parser.add_argument('--lora_model_path', '-lora', type=str, help='LORA模型文件所在路径',
                        default='/workspace/checkpoints/lora/chinese-alpaca-lora-7b')
    parser.add_argument('--device', '-d', help='使用设备，cpu或cuda:0等', type=str, default='cuda:0')
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--host', '-H', type=str, help='监听Host', default='0.0.0.0')
    parser.add_argument('--port', '-P', type=int, help='监听端口号', default=80)
    args = parser.parse_args()

    tokenizer, model = load_llama_tokenizer_and_model(
        args.model_path,
        args.lora_model_path,
        load_8bit=args.load_8bit,
        device=args.device,
    )

    system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    user_prompt = "### Instruction:\n\n{}\n\n### Response:\n\n"
    assistant_prompt = "{}\n\n"
    end_of_text = tokenizer.eos_token
    stop_words = ["### Instruction", "### Response", tokenizer.eos_token]

    uvicorn.run(app, host=args.host, port=args.port)

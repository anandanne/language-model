import argparse
import secrets
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModel, AutoTokenizer

from utils.api import Body, ChatBody, serialize, map_choice


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

    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "text_completion",
        "created": round(time.time()),
        "model": args.model_path,
        "choices": [],
    }

    async def event_generator():
        size = 0
        for response, _ in model.stream_chat(
            tokenizer,
            prompt,
            history=None,
            max_length=max(2048, body.max_tokens),
            top_p=getattr(body, "top_p", 0.7),
            temperature=getattr(body, "temperature", 0.95),
        ):
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
        response, _ = model.chat(
            tokenizer,
            prompt,
            history=None,
            max_length=max(2048, body.max_tokens),
            top_p=getattr(body, "top_p", 0.7),
            temperature=getattr(body, "temperature", 0.95),
        )
        data = template.copy()
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
        "model": args.model_path,
        "choices": [],
    }

    async def chat_event_generator():
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
            data["choices"] = [map_choice(response, delta, chat=True)]

            if await request.is_disconnected():
                return
            yield serialize(data)

        torch_gc()
        yield "[DONE]"

    if body.stream:
        return EventSourceResponse(chat_event_generator())
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
        data["choices"] = [map_choice(response, chat=True)]

        data["usage"] = {
            "prompt_tokens": len(question),
            "completion_tokens": len(response),
            "total_tokens": len(question) + len(response),
        }

        torch_gc()

        return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple API server for ChatGLM-6B')
    parser.add_argument('--model_path', '-m', help='模型文件所在路径', type=str,
                        default='/workspace/checkpoints/chatglm-6b')
    parser.add_argument('--device', '-d', help='使用设备，cpu或cuda:0等', type=str, default='cuda:0')
    parser.add_argument('--quantize', '-q', help='量化等级。可选值：16，8，4', type=int, default=16)
    parser.add_argument('--host', '-H', type=str, help='监听Host', default='0.0.0.0')
    parser.add_argument('--port', '-P', type=int, help='监听端口号', default=80)
    args = parser.parse_args()

    quantize = int(args.quantize)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if args.device == 'cpu':
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True
        ).float()
    else:
        if quantize == 16:
            model = AutoModel.from_pretrained(
                args.model_path,
                trust_remote_code=True,
            ).half().to(args.device)
        else:
            model = AutoModel.from_pretrained(
                args.model_path,
                trust_remote_code=True,
            ).half().quantize(quantize).to(args.device)

    model.eval()

    uvicorn.run(app, host=args.host, port=args.port)

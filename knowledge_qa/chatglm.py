from typing import Optional, List

import openai
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

openai.api_base = "http://192.168.0.53:80/v1"
openai.api_key = "xxx"


class ChatGLM(LLM):

    max_token: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history: list = []
    window_size: int = 10
    use_api: bool = True

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = []
        for hist in self.history[-self.window_size:]:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": hist[0] if hist is not None else ""
                    },
                    {
                        "role": "assistant",
                        "content": hist[1] if hist is not None else ""
                    }
                ]
            )

        messages.append({"role": "user", "content": prompt})
        completion = openai.ChatCompletion.create(
            model="chatglm-6b",
            messages=messages,
            max_tokens=self.max_token,
            temperature=self.temperature,
        )
        response = completion["choices"][0]["message"]["content"]

        if stop is not None:
            response = enforce_stop_tokens(response, stop)

        self.history += [[None, response]]

        return response

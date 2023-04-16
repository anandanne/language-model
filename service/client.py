from langchain.prompts import PromptTemplate

from utils.openai_llm import start_chat_by_chain
from utils.prompt import (
    CHATGLM_PROMPT_TEMPLATE,
    CHINESE_ALPACA_PROMPT_TEMPLATE,
    FIREFLY_PROMPT_TEMPLATE,
    PHOENIX_PROMPT_TEMPLATE,
)

TEMPLATE_MAP = {
    "chatglm": CHATGLM_PROMPT_TEMPLATE,
    "chinese-alpaca": CHINESE_ALPACA_PROMPT_TEMPLATE,
    "firefly": FIREFLY_PROMPT_TEMPLATE,
    "phoenix": PHOENIX_PROMPT_TEMPLATE,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Simple Client for LLMs')
    parser.add_argument('--model_name', '-m', help='模型名称', type=str, default='chatglm')
    parser.add_argument('--api_url', help='api地址', type=str, default="http://192.168.0.53/v1")
    args = parser.parse_args()

    prompt = PromptTemplate(
        input_variables=["history", "input"], template=TEMPLATE_MAP[args.model_name]
    )
    start_chat_by_chain(
        args.model_name,
        args.api_url,
        prompt=prompt,
        verbose=True,
    )

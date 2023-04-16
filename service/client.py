from langchain.prompts import PromptTemplate

from utils.openai_llm import start_chat_by_chain
from utils.prompt import (
    CHATGLM_PROMPT_TEMPLATE,
    CHINESE_ALPACA_PROMPT_TEMPLATE,
    CHATGLM_HUMAN_PREFIX,
    CHINESE_ALPACA_HUMAN_PREFIX,
    CHATGLM_AI_PREFIX,
    CHINESE_ALPACA_AI_PREFIX,
)

TEMPLATE_MAP = {
    "chatglm": (CHATGLM_PROMPT_TEMPLATE, CHATGLM_HUMAN_PREFIX, CHATGLM_AI_PREFIX),
    "chinese-alpaca": (CHINESE_ALPACA_PROMPT_TEMPLATE, CHINESE_ALPACA_HUMAN_PREFIX, CHINESE_ALPACA_AI_PREFIX)
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Simple Client for LLMs')
    parser.add_argument('--model_name', '-m', help='模型名称', type=str, default='chatglm')
    parser.add_argument('--api_url', help='api地址', type=str, default="http://192.168.0.53/v1")
    args = parser.parse_args()

    prompt = PromptTemplate(
        input_variables=["history", "input"], template=TEMPLATE_MAP[args.model_name][0]
    )
    human_prefix, ai_prefix = TEMPLATE_MAP[args.model_name][1], TEMPLATE_MAP[args.model_name][2]
    start_chat_by_chain(
        args.model_name,
        args.api_url,
        prompt=prompt,
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
        verbose=True,
    )

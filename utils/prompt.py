from .registry import BaseParent


class ChatPrompt(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key):
        return cls.registry[class_key.lower()]

    @classmethod
    def __getitem__(cls, key):
        assert (
            key in cls.registry
        ), f"Class {key} not found in base class {cls.__name__} registry {cls.registry}"
        return cls.registry[key]


CHATGLM_PROMPT_TEMPLATE = """{history}
问：{input}
答："""

CHINESE_ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

{history}

### Instruction:

{input}

### Response:

"""

FIREFLY_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

{history}<s>{input}</s></s>"""

PHOENIX_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

{history}Human: <s>{input}</s>Assistant: <s>"""


ChatPrompt.add_to_registry("chatglm", CHATGLM_PROMPT_TEMPLATE)
ChatPrompt.add_to_registry("chinese-alpaca", CHINESE_ALPACA_PROMPT_TEMPLATE)
ChatPrompt.add_to_registry("firefly", FIREFLY_PROMPT_TEMPLATE)
ChatPrompt.add_to_registry("phoenix", PHOENIX_PROMPT_TEMPLATE)

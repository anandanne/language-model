CHATGLM_PROMPT_TEMPLATE = """
Current conversation:
{history}
问: {input}
答: """

CHATGLM_HUMAN_PREFIX = "问"
CHATGLM_AI_PREFIX = "答"

CHINESE_ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

Current conversation:

{history}

### Instruction:

{input}

### Response:

"""

CHINESE_ALPACA_HUMAN_PREFIX = "### Instruction"
CHINESE_ALPACA_AI_PREFIX = "### Response"

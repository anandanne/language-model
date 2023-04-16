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
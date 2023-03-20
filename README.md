# 语言模型

## GPT2

`GPT2` 模型来自 `OpenAI` 的论文[《Language Models are Unsupervised Multitask Learners》](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)无监督的多任务学习语言模型。

## DEMO

有以下两种方式

1. 启动 `Flask DEMO`：

```commandline
# windows
set MODEL=models/gpt2-abstract/checkpoint-358000
set MODEL_LOCAL_FILES_ONLY=yes
set TOKENIZER_NAME=bert

# linux
export MODEL=models/gpt2-abstract/checkpoint-358000
export MODEL_LOCAL_FILES_ONLY=yes
export TOKENIZER_NAME=bert

python -m basaran 
```

2. 启动 `Gradio DEMO`:

```commandline
cd demo/gpt2

python app.py
```
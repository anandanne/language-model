# 语言模型

## 🐼 模型列表

+ `GPT2`：[《Language Models are Unsupervised Multitask Learners》](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

+ `ChatGLM-6b`：[ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型](https://github.com/THUDM/ChatGLM-6B)

+ `LLaMA`：[LLaMA: Open and Efficient Foundation Language Models](https://github.com/facebookresearch/llama)

## PYTORCH镜像

构建镜像

```docker
bash docker/build.sh 
```

启动容器

```docker
bash docker/run.sh
```

进入容器

```docker
docker exec -it llm /bin/bash
```

## 📚 数据集

1. 50 万条中文 `ChatGPT` 指令 `Belle` 数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
2. 100 万条中文 `ChatGPT` 指令 `Belle` 数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
3. 5 万条英文 `ChatGPT` 指令 `Alpaca` 数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
4. 2 万条中文 `ChatGPT` 指令 `Alpaca` 数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
5. 69 万条中文指令 `Guanaco` 数据集(`Belle` 50 万条 + `Guanaco` 19 万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)



## [模型训练](./llmtuning)

针对模型进行纯文本预训练、指令微调和参数高效微调等

## [模型效果](./service)

模型生成效果展示

## [模型评估](./evaluate)

给定任务或指令，评估不同模型的生成效果
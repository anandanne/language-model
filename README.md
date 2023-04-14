# 语言模型

## 模型列表

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

## [模型训练](./llmtuning)

针对模型进行纯文本预训练、指令微调和参数高效微调等

## [模型效果](./service/web)

模型生成效果展示

## [模型评估](./evaluate)

给定任务或指令，评估不同模型的生成效果
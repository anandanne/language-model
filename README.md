# 语言模型

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

| 数据集                                                                          | 数目       | Lang  | Task  | Gen | 类型                                    | 来源                          | 链接                                                                                       |
|:-----------------------------------------------------------------------------|:---------|:------|:------|:----|:--------------------------------------|:----------------------------|:-----------------------------------------------------------------------------------------|
| [Chain of Thought](https://github.com/google-research/FLAN)                  | 74771    | EN/CN | MT    | HG  | CoT相关任务                               | 人在现有数据集上标注CoT               | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chain-of-Thought)     |
| [GPT4all](https://github.com/nomic-ai/gpt4all)                               | 806199   | EN    | MT    | COL | 代码，故事，对话                              | GPT-3.5-turbo 蒸馏            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all)              |
| [GPTeacher](https://github.com/teknium1/GPTeacher)                           | 29013    | EN    | MT    | SI  | 通用，角色扮演，工具指令                          | GPT-4 & toolformer          | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPTeacher)            |
| [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)     | 534610   | ML    | MT    | SI  | 多种nlp任务                               | text-davinci-003            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Guanaco)              |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)                    | 37175    | EN/CN | TS    | MIX | 对话评估                                  | gpt-3.5 或 人工                | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/HC3)                  |
| [alpaca](https://github.com/tatsu-lab/stanford_alpaca)                       | 52002    | EN    | MT    | SI  | 通用指令                                  | text-davinci-003            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca)               |
| [Natural Instructions](https://github.com/allenai/natural-instructions)      | 5040134  | ML    | MT    | COL | 多种nlp任务                               | 人工标注的数据集的收集                 | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Natural-Instructions) |
| [belle_cn](https://huggingface.co/BelleGroup)                                | 1079517  | CN    | TS/MT | SI  | 通用指令，数学推理，对话                          | text-davunci-003            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/belle_cn)             |
| [instinwild](https://github.com/XueFuzhao/InstructionWild)                   | 52191    | EN/CN | MT    | SI  | 生成，开放域问答，头脑风暴                         | text-davunci-003            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instinwild)           |
| [prosocial dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | 165681   | EN    | TS    | MIX | 对话                                    | GPT-3改写问题，人工回复              | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/prosocial-dialog)     |
| [finance_en](https://huggingface.co/datasets/gbharti/finance-alpaca)         | 68912    | EN    | TS    | COL | 金融领域问答                                | GPT3.5                      | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/)                     |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)                        | 78883588 | ML    | MT    | COL | 多种nlp任务                               | 人工标注的数据集的收集                 | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/xP3)                  |
| [firefly](https://github.com/yangjianxin1/Firefly)                           | 1649398  | CN    | MT    | COL | 23种nlp任务                              | 收集中文数据集，人工书写指令模板            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/firefly)              |
| [instruct](https://huggingface.co/datasets/swype/instruct)                   | 888969   | EN    | MT    | COL | GPT4All，Alpaca和开源数据集的增强               | 使用AllenAI提供的nlp增强工具         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instruct)             |
| [Code Alpaca](https://github.com/sahil280114/codealpaca)                     | 20022    | EN    | SI    | SI  | 代码生成，编辑，优化                            | text-davinci-003            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/CodeAlpaca)           |
| [Alpaca_GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)    | 52002    | EN/CN | MT    | SI  | 通用指令                                  | GPT-4 生成的Alpaca数据           | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpacaGPT4)           |
| [webGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)          | 18994    | EN    | TS    | MIX | 信息检索问答                                | fine-tuned GPT-3 + 人工评估     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/webGPT)               |
| [dolly 2.0](https://github.com/databrickslabs/dolly)                         | 15015    | EN    | TS    | HG  | 公开、封闭式问答、信息抽取、摘要生成、开放式构思、分类以及创意写作七类任务 | 人工标注                        | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/dolly)                |
| [baize](https://github.com/project-baize/baize-chatbot)                      | 653699   | EN    | MT    | COL | Alpaca和多种问答任务                         | 人工标注的数据集的收集                 | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/baize)                |
| [hh-rlhf](https://github.com/anthropics/hh-rlhf)                             | 284517   | EN    | TS    | MIX | 对话                                    | RLHF models                 | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/hh-rlhf)              |
| [OIG(part)](https://laion.ai/blog/oig-dataset/)                              | 49237    | EN    | MT    | COL | 多种nlp任务                               | 人工标注的数据集的收集和数据增强            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/OIG)                  |
| [FastChat](https://github.com/lm-sys/FastChat)                               |          | EN    | MT    | MIX | 通用指令                                  | 众包收集ChatGPT与人的交互 (ShareGPT) |                                                                                          |


## 🐼 模型列表

+ `GPT2`：[《Language Models are Unsupervised Multitask Learners》](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

+ `ChatGLM-6b`：[ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型](https://github.com/THUDM/ChatGLM-6B)

+ `LLaMA`：[LLaMA: Open and Efficient Foundation Language Models](https://github.com/facebookresearch/llama)


| Model                                                                   | Backbone |  #Params | Open-source model | Open-source data | Claimed language | Post-training (instruction) | Post-training (conversation) | Release date |
|-------------------------------------------------------------------------|----------|---------:|------------------:|-----------------:|-----------------:|----------------------------:|-----------------------------:|-------------:|
| ChatGPT                                                                 | -        |        - |                 ❌ |                ❌ |            multi |                             |                              |     11/30/22 |
| Wenxin                                                                  | -        |        - |                 ❌ |                ❌ |               zh |                             |                              |     03/16/23 |
| [ChatGLM](https://github.com/THUDM/ChatGLM-6B)                          | GLM      |       6B |                 ✅ |                ❌ |           en, zh |                             |                              |     03/16/23 |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                  | LLaMA    |       7B |                 ❌ |                ✅ |               en |                     52K, en |                            ❌ |     03/13/23 |
| [Dolly](https://github.com/databrickslabs/dolly)                        | GPT-J    |       6B |                 ✅ |                ✅ |               en |                     52K, en |                            ❌ |     03/24/23 |
| [BELLE](https://github.com/LianjiaTech/BELLE)                           | BLOOMZ   |       7B |                 ✅ |                ✅ |               zh |                    1.5M, zh |                            ❌ |     03/26/23 |
| [Guanaco](https://guanaco-model.github.io/)                             | LLaMA    |       7B |                 ✅ |                ✅ |   en, zh, ja, de |                 534K, multi |                            ❌ |     03/26/23 |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)   | LLaMA    |    7/13B |                 ✅ |                ✅ |           en, zh |                2M/3M, en/zh |                            ❌ |     03/28/23 |
| [LuoTuo](https://github.com/LC1332/Luotuo-Chinese-LLM)                  | LLaMA    |       7B |                 ✅ |                ✅ |               zh |                     52K, zh |                            ❌ |     03/31/23 |
| [Vicuna](https://github.com/lm-sys/FastChat)                            | LLaMA    |    7/13B |                 ✅ |                ✅ |               en |                           ❌ |                   70K, multi |     03/13/23 |
| [Koala](https://github.com/young-geng/EasyLM)                           | LLaMA    |      13B |                 ✅ |                ✅ |               en |                    355K, en |                     117K, en |     04/03/23 |
| [BAIZE](https://github.com/project-baize/baize-chatbot)                 | LLaMA    | 7/13/30B |                 ✅ |                ✅ |               en |                           ❌ |                   111.5K, en |     04/04/23 |
| [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)                | BLOOMZ   |       7B |                 ✅ |                ✅ |            multi |                         40+ |                          40+ |     04/08/23 |
| [Latin Phoenix: Chimera](https://github.com/FreedomIntelligence/LLMZoo) | LLaMA    |    7/13B |                 ✅ |                ✅ |    multi (Latin) |                       Latin |      


## [模型训练](./llmtuning)

针对模型进行纯文本预训练、指令微调和参数高效微调等

## [模型效果](./service)

模型生成效果展示

## [模型评估](./evaluate)

给定任务或指令，评估不同模型的生成效果
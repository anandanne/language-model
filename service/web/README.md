# 模型对话界面

使用**接口方式**

## 1. 启动接口

### ChatGLM-6B

```shell
python service/apis/chatglm.py
```

### LLaMA

```shell
python service/apis/llama.py
```

## 2. 启动对话界面

```shell
export CHATGLM_COMPLETION_URL=http://192.168.0.53:80/v1/chat/completions
export LLAMA_COMPLETION_URL=http://192.168.0.59:80/v1/chat/completions

cd service

python web/chatbot.py
```

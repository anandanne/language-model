# 启动对话界面

使用**接口方式**

### ChatGLM-6B

首先启动模型接口

```commandline
python service/apis/chatglm.py
```

然后启动对话界面

```commandline
export CHATGLM_COMPLETION_URL=http://192.168.0.53:80/v1/chat/completions

cd service

python web/chatbot.py
```

### LLaMA

首先启动模型接口

```commandline
python service/apis/llama.py
```

然后启动对话界面

```commandline
export LLAMA_COMPLETION_URL=http://192.168.0.59:80/v1/chat/completions

cd service

python web/chatbot.py
```
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

### Bloom

```shell
python service/apis/firefly.py
```

## 2. 启动对话界面

```shell
export CHATGLM_COMPLETION_URL=http://192.168.0.53:80/v1/chat/completions
export LLAMA_COMPLETION_URL=http://192.168.0.59:80/v1/chat/completions
# export FIREFLY_COMPLETION_URL=http://192.168.0.59:80/v1/chat/completions

cd service

python web/chatbot.py
```

# 命令端启动方式

```shell
python service/client.py
```

可选参数：

+ `model_name`： `chatglm` 或者 `chinese-alpaca`， 默认为 `chatglm`

+ `api_url`：接口地址，默认为 `http://192.168.0.53/v1`

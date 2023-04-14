
以下命令均在项目根目录下启动

## ChatGLM

```shell
python service/apis/chatglm.py
```

其中 `model_path` 为下载好的模型所在路径

## LLaMA

```shell
python service/apis/chinese_alpaca.py
```

根据需要更改基础模型和 `lora` 模型权重的路径

## Firefly-2b6

```shell
python service/apis/firefly.py
```

其中 `model_path` 为下载好的模型所在路径
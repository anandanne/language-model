## ChatGLM

```shell
python service/apis/chatglm.py
```

可选参数：

+ `model_path`： `ChatGLM` 模型文件所在路径， 默认为 `/workspace/checkpoints/chatglm-6b`

+ `device`：使用设备，`cpu` 或 `cuda:0` 等，默认为 `cuda:0`

+ `quantize`：量化等级，可选值：`16，8，4`，默认为 `16`

+ `host`：监听Host，默认为 `0.0.0.0`

+ `port`：监听Port，默认为 `80`

## LLaMA

```shell
python service/apis/chinese_alpaca.py
```

可选参数：

+ `model_path`： `LLaMA` 模型文件所在路径， 默认为 `/workspace/checkpoints/llama-7b-hf`

+ `lora_model_path`： `LORA` 模型文件所在路径， 默认为 `/workspace/checkpoints/llama-7b-hf`

+ `device`：使用设备，`cpu` 或 `cuda:0` 等，默认为 `cuda:0`

+ `load_8bit`：使用 `8bit` 量化，默认为 `False`

+ `host`：监听Host，默认为 `0.0.0.0`

+ `port`：监听Port，默认为 `80`

## Firefly-2b6

```shell
python service/apis/firefly.py
```

可选参数：

+ `model_path`： 模型文件所在路径， 默认为 `/workspace/checkpoints/firefly-2b6`

+ `device`：使用设备，`cpu` 或 `cuda:0` 等，默认为 `cuda:0`

+ `host`：监听Host，默认为 `0.0.0.0`

+ `port`：监听Port，默认为 `80`
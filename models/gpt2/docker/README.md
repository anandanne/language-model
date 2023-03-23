# Docker训练

单机单卡训练

```commandline
sudo docker run --name gpt -it -d --gpus=all \
    -v /home/xusenlin/Projects/NLP/GPT/docker/data:/workspace/data \
    -v /home/xusenlin/Projects/NLP/GPT/docker/model:/workspace/model \
    -v /home/xusenlin/Projects/NLP/GPT/docker/run_gpt2_chinese_abstract.sh:/workspace/run_gpt2_chinese_abstract.sh \
    transformers:gpu /bin/bash run_gpt2_chinese_abstract.sh
```

单机多卡训练

```commandline
sudo docker run --name gpt -it -d --gpus=all \
    -v /home/xusenlin/Projects/NLP/GPT/docker/data:/workspace/data \
    -v /home/xusenlin/Projects/NLP/GPT/docker/model:/workspace/model \
    -v /home/xusenlin/Projects/NLP/GPT/docker/run_gpt2_chinese_abstract.sh:/workspace/run_gpt2_chinese_abstract.sh \
    transformers:gpu /bin/bash run_gpt2_chinese_abstract_ddp.sh
```

部分可配置参数含义：

+ `CUDA_VISIBLE_DEVICES`: 使用的 `GPU` 序号

+ `nproc_per_node`: 单个机器的 `GPU` 数量

+ `model_name_or_path`：模型文件路径，包含配置、权重、词表等

+ `train_file`：训练集文件名或文件名列表

+ `validation_file`：验证集文件名或文件名列表

+ `cache_dir`：数据和模型的缓存路径

+ `per_device_train_batch_size`：训练时单个 `gpu` 设备的批量大小

+ `per_device_eval_batch_size`：验证时单个 `gpu` 设备的批量大小

+ `num_train_epochs`：训练轮次

+ `save_total_limit`：最多保存的模型个数

+ `do_train`：是否进行训练

+ `do_eval`：是否进行验证评估

+ `save_steps`：每隔多少步保存一次模型

+ `logging_steps`：每隔多少步打印一次日志

+ `output_dir`：模型结果保存路径

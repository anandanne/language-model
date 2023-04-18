## 单卡训练

### LLaMA

```shell
python uniform_lora_finetune.py \
    --model_type llama \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data alpaca-belle-cot \
    --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 \
    --learning_rate 3e-4 \
    --epochs 1 
```  
  
  
### ChatGLM

```shell
python uniform_lora_finetune.py \
    --model_type chatglm \
    --model_name_or_path THUDM/chatglm-6b \
    --data alpaca-belle-cot \
    --lora_target_modules query_key_value \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_gpu_train_batch_size 2 \
    --learning_rate 2e-5 \
    --epochs 1
```


### BLOOM

```shell
python uniform_lora_finetune.py \
    --model_type bloom \
    --model_name_or_path bigscience/bloomz-7b1-mt \
    --data alpaca-belle-cot \
    --lora_target_modules query_key_value \
    --per_gpu_train_batch_size 4 \
    --learning_rate 3e-4 \
    --epochs 1 
```

## 多卡训练

### LLaMA

```shell
python -m torch.distributed.launch \
    --nproc_per_node 4  \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=xxx \
    --master_port=yyy \
    uniform_lora_finetune.py \
    --model_type llama \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data alpaca-belle-cot \
    --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 \
    --learning_rate 3e-4 \
    --epochs 1 
```

### ChatGLM
```shell
python3 -m torch.distributed.launch \
    --nproc_per_node 4  \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=xxx \
    --master_port=yyy \
    uniform_lora_finetune.py \
    --model_type chatglm \
    --model_name_or_path THUDM/chatglm-6b \
    --data alpaca-belle-cot \
    --lora_target_modules query_key_value \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_gpu_train_batch_size 2 \
    --learning_rate 2e-5 \
    --epochs 1
```

### BLOOM
```shell
python -m torch.distributed.launch \
  --nproc_per_node 4  \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=xxx \
    --master_port=yyy \
    uniform_lora_finetune.py \
    --model_type bloom \
    --model_name_or_path bigscience/bloomz-7b1-mt \
    --data alpaca-belle-cot \
    --lora_target_modules query_key_value \
    --per_gpu_train_batch_size 4 \
    --learning_rate 3e-4 \
    --epochs 1  
```

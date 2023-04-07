export NCCL_SOCKET_IFNAME="eth0,eth,ib,eno1,enp4s0"
export MASTER_ADDR="192.168.0.59"
export MASTER_PORT="29500"

torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_clm.py \
    --config_name checkpoints/gpt2-chinese \
    --tokenizer_name checkpoints/gpt2-chinese \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --cache_dir data \
    --streaming \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 700_000 \
    --save_total_limit 4 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --lr_scheduler_type cosine \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --warmup_steps 1_000 \
    --eval_steps 5_000 \
    --save_steps 5_000 \
    --logging_steps 500 \
    --load_best_model_at_end \
    --output_dir outputs/gpt2-abstract
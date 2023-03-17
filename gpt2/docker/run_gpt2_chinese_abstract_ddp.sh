CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node 7 run_gpt2_chinese_abstract.py \
    --config_name model/config.json \
    --tokenizer_name model/gpt-cpm-cn-sentencepiece.model \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --cache_dir data \
    --streaming \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 500_000 \
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
    --output_dir outputs/gpt2-chinese

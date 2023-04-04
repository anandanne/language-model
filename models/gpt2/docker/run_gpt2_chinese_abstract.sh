#! /bin/bash
python run_gpt2_chinese_abstract.py \
    --config_name model/config.json \
    --tokenizer_name model/gpt-cpm-cn-sentencepiece.model \
    --tokenizer_type cpm \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --cache_dir data \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 20 \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --save_steps 1000 \
    --logging_steps 1000 \
    --output_dir outputs/gpt2-chinese
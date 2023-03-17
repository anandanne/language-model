CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node 7 run_gpt2_chinese_abstract.py \
    --config_name model/config.json \
    --tokenizer_name model/gpt-cpm-cn-sentencepiece.model \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --cache_dir data \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --save_steps 5000 \
    --logging_steps 5000 \
    --output_dir outputs/gpt2-chinese

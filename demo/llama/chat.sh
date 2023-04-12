CUDA_VISIBLE_DEVICES=0 \
  deepspeed demo/llama/chat.py \
      --deepspeed demo/configs/ds_config_chatbot.json \
      --model_name_or_path decapoda-research/llama-7b-hf \
      --cache_dir checkpoints \
      --lora_model_path lora/llama7b-lora-380k /
sudo docker run --name gpt2 -it -d --gpus=all \
    -v /home/xusenlin/Projects/NLP/GPT/docker/data:/workspace/data \
    -v /home/xusenlin/Projects/NLP/GPT/docker/model:/workspace/model \
    -v /home/xusenlin/Projects/NLP/GPT/docker/tokenization.py:/workspace/tokenization.py \
    -v /home/xusenlin/Projects/NLP/GPT/docker/run_gpt2_chinese_abstract.py:/workspace/run_gpt2_chinese_abstract.py \
    -v /home/xusenlin/Projects/NLP/GPT/docker/run_gpt2_chinese_abstract.sh:/workspace/run_gpt2_chinese_abstract.sh \
    transformers:gpu /bin/bash run_gpt2_chinese_abstract.sh
sudo docker run --name gpt2 -it -d --gpus=all --ipc=host \
    -v /home/luser/gpt2/docker/data:/workspace/data \
    -v /home/luser/gpt2/docker/model:/workspace/model \
    -v /home/luser/gpt2/docker/tokenization.py:/workspace/tokenization.py \
    -v /home/luser/gpt2/docker/run_gpt2_chinese_abstract.py:/workspace/run_gpt2_chinese_abstract.py \
    -v /home/luser/gpt2/docker/run_gpt2_chinese_abstract.sh:/workspace/run_gpt2_chinese_abstract.sh \
    -v /home/luser/gpt2/docker/run_gpt2_chinese_abstract_ddp.sh:/workspace/run_gpt2_chinese_abstract_ddp.sh \
    transformers:gpu /bin/bash run_gpt2_chinese_abstract_ddp.sh
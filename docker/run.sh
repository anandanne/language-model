docker run -it -d --gpus all --ipc=host -p 80:80 -p:7860:7860 --name=llm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e MODEL=checkpoints/gpt2-abstract/checkpoint-500000 \
    -e SERVER_MODEL_NAME=gpt2-abstract \
    -e MODEL_LOCAL_FILES_ONLY=true \
    -e TOKENIZER_NAME=bert \
    -e REMOVE_WHITESPACE=true \
    -v `pwd`:/workspace \
    pytorch-llm:gpu
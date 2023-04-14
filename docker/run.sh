docker run -it -d --gpus all --ipc=host --net=host -p 80:80 -p:7860:7860 --name=llm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm:pytorch
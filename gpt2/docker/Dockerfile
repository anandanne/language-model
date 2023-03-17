FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install transformers==4.26.1 && \
    pip install datasets && \
    pip install scikit-learn && \
    pip install accelerate && \
    pip install protobuf && \
    pip install jieba && \
    pip install sentencepiece

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
RUN mkdir -p /workspace/data
COPY . /workspace
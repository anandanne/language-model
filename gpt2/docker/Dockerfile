FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install transformers==4.26.1 && \
    pip install datasets==2.10.1 && \
    pip install scikit-learn==1.2.2 && \
    pip install accelerate>=0.12.0 && \
    pip install protobuf && \
    pip install jieba==0.42.1 && \
    pip install sentencepiece==0.1.97 && \
    pip cache purge

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
RUN mkdir -p /workspace/data
COPY . /workspace
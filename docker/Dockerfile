FROM nvcr.io/nvidia/pytorch:23.02-py3

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
COPY requirements.txt /workspace/

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt -i  https://mirror.baidu.com/pypi/simple

import re

import markdown
from latex2mathml.converter import convert as tex2mathml

incomplete = '<font style="color:orange;" class="tooltip">&#9888;<span class="tooltiptext">formula incomplete</span></font>'
convError = '<font style="color:red" class="tooltip">&#9888;<span class="tooltiptext">LaTeX-convert-error</span></font>'


def convert(mdtex, extensions=[], splitParagraphs=True):
    """ converts recursively the Markdown-LaTeX-mixture to HTML with MathML """
    found = False
    # handle all paragraphs separately (prevents aftereffects)
    if splitParagraphs:
        parts = re.split("\n\n", mdtex)
        result = ''
        for part in parts:
            result += convert(part, extensions, splitParagraphs=False)
        return result
    # find first $$-formula:
    parts = re.split('\${2}', mdtex, 2)
    if len(parts) > 1:
        found = True
        result = convert(parts[0], extensions, splitParagraphs=False) + '\n'
        try:
            result += '<div class="blockformula">' + tex2mathml(parts[1]) + '</div>\n'
        except:
            result += '<div class="blockformula">' + convError + '</div>'
        if len(parts) == 3:
            result += convert(parts[2], extensions, splitParagraphs=False)
        else:
            result += '<div class="blockformula">' + incomplete + '</div>'
    # else find first $-formulas:
    else:
        parts = re.split('\${1}', mdtex, 2)
    if len(parts) > 1 and not found:
        found = True
        try:
            mathml = tex2mathml(parts[1])
        except:
            mathml = convError
        if parts[0].endswith('\n\n') or parts[0] == '':  # make sure textblock starts before formula!
            parts[0] = parts[0] + '&#x200b;'
        if len(parts) == 3:
            result = convert(parts[0] + mathml + parts[2], extensions, splitParagraphs=False)
        else:
            result = convert(parts[0] + mathml + incomplete, extensions, splitParagraphs=False)
    # else find first \[..\]-equation:
    else:
        parts = re.split(r'\\\[', mdtex, 1)
    if len(parts) > 1 and not found:
        found = True
        result = convert(parts[0], extensions, splitParagraphs=False) + '\n'
        parts = re.split(r'\\\]', parts[1], 1)
        try:
            result += '<div class="blockformula">' + tex2mathml(parts[0]) + '</div>\n'
        except:
            result += '<div class="blockformula">' + convError + '</div>'
        if len(parts) == 2:
            result += convert(parts[1], extensions, splitParagraphs=False)
        else:
            result += '<div class="blockformula">' + incomplete + '</div>'
    # else find first \(..\)-equation:
    else:
        parts = re.split(r'\\\(', mdtex, 1)
    if len(parts) > 1 and not found:
        found = True
        subp = re.split(r'\\\)', parts[1], 1)
        try:
            mathml = tex2mathml(subp[0])
        except:
            mathml = convError
        if parts[0].endswith('\n\n') or parts[0] == '':  # make sure textblock starts before formula!
            parts[0] = parts[0] + '&#x200b;'
        if len(subp) == 2:
            result = convert(parts[0] + mathml + subp[1], extensions, splitParagraphs=False)
        else:
            result = convert(parts[0] + mathml + incomplete, extensions, splitParagraphs=False)
    if not found:
        result = mdtex
    return result


def regular_txt_to_markdown(text):
    """
    将普通文本转换为Markdown格式的文本。
    """
    text = text.replace('\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    return text


def text_divide_paragraph(text):
    """
    将文本按照段落分隔符分割开，生成带有段落标签的HTML代码。
    """
    if '```' in text:
        # careful input
        return text
    else:
        # wtf input
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i != 0: lines[i] = "<p>" + lines[i].replace(" ", "&nbsp;") + "</p>"
        text = "".join(lines)
        return text


def markdown_convertion(txt):
    """
    将Markdown格式的文本转换为HTML格式。如果包含数学公式，则先将公式转换为HTML格式。
    """
    if ('$' in txt) and ('```' not in txt):
        return markdown.markdown(txt, extensions=['fenced_code', 'tables']) + '<br><br>' + \
               markdown.markdown(convert(txt, splitParagraphs=False), extensions=['fenced_code', 'tables'])
    else:
        return markdown.markdown(txt, extensions=['fenced_code', 'tables'])


def format_io(self, y):
    """
    将输入和输出解析为HTML格式。将y中最后一项的输入部分段落化，并将输出部分的Markdown和数学公式转换为HTML格式。
    """
    if y is None: return []
    i_ask, gpt_reply = y[-1]
    y[-1] = (
        None if i_ask is None else i_ask,
        None if gpt_reply is None else markdown_convertion(gpt_reply)
    )
    return y


# docker run -it -p 29500:29500 \
#     --name colossal_llm \
#     --gpus all \
#     -v /home/xusenlin/NLP/ColossalAI/:/workspace/ColossalAI/ \
#     colossalai:0.2.8 \
#     /bin/bash
#
# docker run -it --rm -p 29500:29500 \
#     -v /home/xusenlin/NLP/ColossalAI/test.py:/home/test.py \
#     --net=host docker.io/pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda11.7.1 \
#     python /home/test.py --rank 0 --world-size 2 --init-method tcp://192.168.0.59:29500 --backend gloo
#
#
# docker run -it --rm -p 29500:29500 \
#     -v /home/xusenlin/NLP/ColossalAI/test.py:/home/test.py \
#     --net=host docker.io/pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda11.7.1 \
#     python /home/test.py --rank 1 --world-size 2 --init-method tcp://192.168.0.59:29500 --backend gloo
#
#
# docker run -it -p 29500:29500 \
#     --gpus all --name pl \
#     -v /home/xusenlin/NLP/ColossalAI/test.py:/home/test.py \
#     docker.io/pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda11.7.1
#
# export MASTER_ADDR="192.168.0.59"
# export MASTER_PORT="29500"
# export LOGLEVEL="DEBUG"
# torchrun \
#     --nnodes=2 \
#     --node_rank=0 \
#     --nproc_per_node=1 \
#     --master_addr="192.168.0.59" \
#     --master_port=29500 \
#     test.py
#
# export MASTER_ADDR="192.168.0.59"
# export MASTER_PORT="29500"
# export LOGLEVEL="DEBUG"
# torchrun \
#     --nnodes=2 \
#     --node_rank=1 \
#     --nproc_per_node=1 \
#     --master_addr="192.168.0.59" \
#     --master_port=29500 \
#     test.py
#
# export MASTER_PORT=29500
# export MASTER_ADDR=192.168.0.59
# export WORLD_SIZE=2
# export LOCAL_WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0
# export LOGLEVEL="DEBUG"
# torchrun test.py
#
# export MASTER_PORT=29500
# export MASTER_ADDR=192.168.0.59
# export WORLD_SIZE=2
# export LOCAL_WORLD_SIZE=1
# export RANK=1
# export LOCAL_RANK=0
# export LOGLEVEL="DEBUG"
# torchrun test.py
#
#
# lightning run model --accelerator=cuda --devices=1 --num_nodes=2 --node_rank=0 --main_address=192.168.0.59 --main_port=29500 test.py
# lightning run model --accelerator=cuda --devices=1 --num_nodes=2 --node_rank=1 --main_address=192.168.0.59 --main_port=29500 test.py
#
# --master_addr="192.168.0.59" --master_port=29500 \
#
# export PYTHONPATH=. && torchrun --nnodes 2 --nproc_per_node 1  --node_rank=0 \
#     --master_addr="192.168.0.59" --master_port=29500 \
#     benchmarks/benchmark_gpt_dummy.py --model s --strategy ddp --train_batch_size 2
#
# export PYTHONPATH=. && torchrun --nnodes 2 --nproc_per_node 1  --node_rank=1 \
#     --master_addr="192.168.0.59" --master_port=29500 \
#     benchmarks/benchmark_gpt_dummy.py --model s --strategy ddp --train_batch_size 2
#
# python -m torch.distributed.launch --nproc_per_node=1 \
#        --nnodes=2 --node_rank=0 --master_addr="192.168.0.59" \
#        --master_port=29500 test.py
#
# python -m torch.distributed.launch --nproc_per_node=1 \
#        --nnodes=2 --node_rank=1 --master_addr="192.168.0.59" \
#        --master_port=29500 test.py
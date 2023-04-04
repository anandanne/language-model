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

"""
Functions for creating and merging choice objects.
"""


def map_choice(
    text,
    index,
    token=None,
    token_logprob=None,
    top_logprobs=None,
    text_offset=None,
    finish_reason=None,
    chat=False,
):
    """Create a choice object from model outputs."""
    choice = {
        "index": index,
        "logprobs": None,
        "finish_reason": finish_reason,
    }

    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text

    # Include log probabilities of the selected and most likely tokens.
    if (
        token is not None
        and token_logprob is not None
        and top_logprobs is not None
        and text_offset is not None
    ):
        choice["logprobs"] = {
            "tokens": [token],
            "token_logprobs": [token_logprob],
            "top_logprobs": [top_logprobs],
            "text_offset": [text_offset],
        }

    return choice


def reduce_choice(choices, chat=False):
    """Merge a list of choices into a single choice object."""
    buffer = []
    index = 0
    finish_reason = None
    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []

    # All choice objects are expected to have the same shape.
    for choice in choices:
        buffer.append(choice["text"])
        index = choice["index"]
        finish_reason = choice["finish_reason"]
        logprobs = choice["logprobs"]
        if logprobs is not None:
            tokens += logprobs["tokens"]
            token_logprobs += logprobs["token_logprobs"]
            top_logprobs += logprobs["top_logprobs"]
            text_offset += logprobs["text_offset"]

    # Create reduced object with the last seen index and finish reason.
    text = "".join(buffer)
    reduced = {
        "index": index,
        "logprobs": None,
        "finish_reason": finish_reason,
    }
    if chat:
        reduced["message"] = {"role": "assistant", "content": text}
    else:
        reduced["text"] = text

    if tokens:
        reduced["logprobs"] = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": text_offset,
        }

    return reduced

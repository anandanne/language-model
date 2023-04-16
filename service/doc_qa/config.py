import torch


EMBEDDING_MODEL_MAP = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "checkpoints/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LLM URL
LLM_MODEL = "chatglm-6b"
CHATGLM_6B_URL = "http://192.168.0.53/v1"

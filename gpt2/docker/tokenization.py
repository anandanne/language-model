"""Tokenization classes for ChatGLM."""
import os
from typing import List, Optional, Union

import jieba
import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model"}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
}


class CpmTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=False,
        unk_token="<unk>",
        bos_token="<bod>",
        eos_token="<eod>",
        eol_token="\u2583",
        pad_token="<eod>",
        padding_side="right",
        **kwargs
    ) -> None:
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            padding_side=padding_side,
            **kwargs
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.vocab_file = vocab_file

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(vocab_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    @property
    def eol_token_id(self):
        if self.eol_token is None:
            return None
        return self.convert_tokens_to_ids(self.eol_token)

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return len(self.sp)

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs

        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text):
        text = self.preprocess_text(text)
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)

        return self.sp.encode(new_seg, out_type=str)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp.IdToPiece(index)

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        if isinstance(token_ids[0], list):
            tokens = []
            for single_token_ids in token_ids:
                if self.pad_token_id in single_token_ids:  # remove pad
                    single_token_ids = list(filter(self.pad_token_id.__ne__, single_token_ids))
                tokens.append(self.sp.decode(single_token_ids))
        else:
            if self.pad_token_id in token_ids:  # remove pad
                token_ids = list(filter(self.pad_token_id.__ne__, token_ids))
            tokens = self.sp.decode(token_ids)

        return tokens.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return vocab_file,

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is not None:
            token_ids_0 += token_ids_1

        token_ids_0 += [self.sp.PieceToId(self.eos_token)]

        return token_ids_0


if __name__ == "__main__":
    tokenizer = CpmTokenizer.from_pretrained("model/gpt-cpm-cn-sentencepiece.model")
    print(tokenizer.vocab_size)
    print(tokenizer.tokenize("欢迎使用百度飞浆！"))
    print(tokenizer.tokenize("如何训练chatgpt?"))

    print(tokenizer("欢迎使用百度飞浆！", add_special_tokens=False))
    print(tokenizer(["欢迎使用百度飞浆！", "如何训练chatgpt?"], padding=True))

    print(tokenizer.batch_decode([[2092, 260, 1014, 1596, 5331, 45, 7, 7], [167, 1557, 9291, 5199, 1129, 14732, 20, 7]]))

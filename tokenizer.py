# -*- coding: utf-8 -*-
r""" 
Text Tokenizer
==============
    Wrapper around GPT2 tokenizer.
"""
import torch
from torchnlp.encoders.text.text_encoder import TextEncoder
from transformers import AutoTokenizer

SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",  
    "pad_token": "<pad>",
}

class Tokenizer(TextEncoder):
    """Wrapper around Hugging-face Auto-tokenizer.

    :param pretrained_model: Transformer pretrained model.
    """

    def __init__(self, pretrained_model, context) -> None:
        self.enforce_reversible = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        orig_vocab = self.tokenizer.vocab_size
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.vocab_size = orig_vocab + num_added_tokens

        self.context = context

        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.sep_token_id
        self.vocab = self.tokenizer.get_vocab()

        if self.context:
            self.bos_index = self.tokenizer.cls_token_id 
        else:
            self.bos_index = self.tokenizer.sep_token_id  


    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        :return: torch.Tensor with Encoding of the `sequence`.
        """
        sequence = TextEncoder.encode(self,sequence)

        if self.context:
            return self.tokenizer(sequence, truncation=True, max_length=256, add_special_tokens=False)["input_ids"]

        else:
            return self.tokenizer(sequence, truncation=True, max_length=256)["input_ids"]
       

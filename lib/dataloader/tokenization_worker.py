import numpy as onp
from transformers import PreTrainedTokenizer
from typing import List, Tuple

from ..preprocessing.distort_sentence import distort_sentence
from ..random.wrapper import KeyArray, split_key

def tokenization_worker_inner(tokenizer: PreTrainedTokenizer, sentences: List[str], key: KeyArray) -> onp.ndarray:
    keys = split_key(key, num=len(sentences))
    distorted_sentences = [distort_sentence(sentence, key=key) for sentence, key in zip(sentences, keys)]

    x = tokenizer(sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)
    y = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)

    src = x.input_ids.astype(onp.uint32)
    mask_enc_1d = x.attention_mask.astype(onp.bool_)
    dst = y.input_ids.astype(onp.uint32)
    mask_dec_1d = y.attention_mask.astype(onp.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d

def tokenization_worker(tokenizer: PreTrainedTokenizer, sentences_key: Tuple[List[str], KeyArray]) -> onp.ndarray:
    sentences, key = sentences_key
    return tokenization_worker_inner(tokenizer, sentences, key)

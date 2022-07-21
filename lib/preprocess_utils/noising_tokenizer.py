import itertools
from typing import List, Tuple

from transformers import PreTrainedTokenizer
import jax.random as rand
import numpy as np

def is_word_start(s: str) -> bool:
    return s.startswith('Ġ')

def group_subwords(xs: List[str]) -> List[List[str]]:
    res = []

    curr_arr = []
    res.append(curr_arr)

    for x in xs:
        if is_word_start(x):
            curr_arr = [x]
            res.append(curr_arr)
        else:
            curr_arr.append(x)
    return res

def tokenize_and_distort_sentence_inner(key: rand.KeyArray, tokenizer: PreTrainedTokenizer, sentence: str, distort_rate: float=0.15) -> List[str]:
    tokenized_sentence = tokenizer.tokenize(sentence)
    words = group_subwords(tokenized_sentence)
    n_words = len(words)
    masks = rand.uniform(key, (n_words,)) < distort_rate
    masked_words = [[tokenizer.mask_token] if mask else word for word, mask in zip(words, masks)]
    return list(itertools.chain.from_iterable(masked_words))

def tokenize_and_distort_sentence(key: rand.KeyArray, tokenizer: PreTrainedTokenizer, sentence: str, sequence_len: int) -> Tuple[List[int], List[bool]]:
    masked_sentence = tokenize_and_distort_sentence_inner(key, tokenizer, sentence)
    ids = tokenizer.convert_tokens_to_ids(masked_sentence)
    if len(ids) > sequence_len - 2:  # 2: <s> and </s>
        ids = ids[:sequence_len - 2]
    n_pads = sequence_len - 2 - len(ids)
    padded_ids = [
        tokenizer.bos_token_id,
        *ids,
        tokenizer.eos_token_id,
        *((tokenizer.pad_token_id,) * n_pads),
    ]
    mask = [
        *((True,) * (sequence_len - n_pads)),
        *((False,) * n_pads),
    ]
    assert len(padded_ids) == sequence_len
    assert len(mask) == sequence_len
    return padded_ids, mask

def tokenize_and_distort_sentences(key: rand.KeyArray, tokenizer: PreTrainedTokenizer, sentences: List[str], sequence_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n_sents = len(sentences)
    keys = rand.split(key, num=n_sents)

    src = []
    mask_1d = []
    for key, sentence in zip(keys, sentences):
        padded_ids, mask = tokenize_and_distort_sentence(key, tokenizer, sentence, sequence_len)
        src.append(padded_ids)
        mask_1d.append(mask)

    src = np.array(src, dtype=np.int32)
    mask_1d = np.array(mask_1d, dtype=np.bool_)
    packed_mask_1d = np.packbits(mask_1d, axis=1)

    return src, packed_mask_1d

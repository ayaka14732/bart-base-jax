from glob import glob
import json
from multiprocessing import Pool
from typing import List, Optional, Tuple

import blingfire
import jax
import jax.random as rand
import numpy as np
from transformers import BartTokenizer

from lib.preprocess_utils.noising_tokenizer import tokenize_and_distort_sentences

from multiprocessing import set_start_method

jax.config.update('jax_platform_name', 'cpu')

n_cpu: Optional[int] = 96
sequence_len: int = 15

key = rand.PRNGKey(42)

def article_to_sentences(text: str) -> List[str]:
    '''
    ```python
    >>> article_to_sentences('A cat. The mouse.')
    ['A cat.', 'The mouse.']
    >>> article_to_sentences('A long line\nwith wrapping. The mouse.')
    ['A long line with wrapping.', 'The mouse.']
    >>> article_to_sentences('\n    ')
    []
    ```
    '''
    if not text.strip():
        return []
    return blingfire.text_to_sentences(text).split('\n')

def filename_to_sentences(filename: str) -> List[str]:
    all_sentences = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text']
            sentences = article_to_sentences(text)
            all_sentences.extend(sentences)
    return all_sentences

def tokenize_sentences(tokenizer, sentences):
    batch = tokenizer(sentences, max_length=sequence_len, padding='max_length', truncation=True, return_tensors='np')
    src = batch.input_ids
    mask_1d = batch.attention_mask.astype(np.bool_)
    return src, mask_1d

def pipeline(key, filename):  # TODO: find a better name
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    sentences = filename_to_sentences(filename)
    tokenized_sentences = tokenize_sentences(tokenizer, sentences)
    noised_sentences = tokenize_and_distort_sentences(key, tokenizer, sentences, sequence_len)
    return tokenized_sentences, noised_sentences

def unzip(list_of_arrs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    a = []
    b = []
    for x, y in list_of_arrs:
        a.append(x)
        b.append(y)
    a = np.vstack(a)
    b = np.vstack(b)
    return a, b

def go(key: rand.KeyArray):
    filenames = glob('./dump/*/*')
    keys = rand.split(key, num=len(filenames))

    with Pool(processes=n_cpu) as p:
        # TODO Debug
        # tokenized_sentences, noised_sentences = p.starmap(pipeline, zip(keys, filenames))
        x = p.starmap(pipeline, zip(keys, filenames))
        tokenized_sentences, noised_sentences, *v = x
        assert not v, v

    dst, mask_dec_1d = unzip(tokenized_sentences)
    src, mask_enc_1d = unzip(noised_sentences)

    return src, mask_enc_1d, dst, mask_dec_1d

if __name__ == '__main__':
    set_start_method("spawn")

    key, subkey = rand.split(key, num=2)
    src, mask_enc_1d, dst, mask_dec_1d = go(subkey)

    print(src.shape)
    print(mask_enc_1d.shape)
    print(dst.shape)
    print(mask_dec_1d.shape)

    print(src.dtype)
    print(mask_enc_1d.dtype)
    print(dst.dtype)
    print(mask_dec_1d.dtype)


    # filename = glob('./dump/*/*')[2]
    # pipeline(key, filename)

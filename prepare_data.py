from glob import glob
import json
from multiprocessing import Pool, set_start_method
from typing import List

import blingfire
import jax
import jax.random as rand
import numpy as np
from transformers import BartTokenizer

from lib.preprocess_utils.noising_tokenizer import tokenize_and_distort_sentences

jax.config.update('jax_platform_name', 'cpu')

sequence_len: int = 256
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
    src = batch.input_ids.astype(np.int32)
    mask_1d = batch.attention_mask.astype(np.bool_)
    return src, mask_1d

def pipeline(key, filename):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    sentences = filename_to_sentences(filename)
    dst, mask_dec_1d = tokenize_sentences(tokenizer, sentences)
    src, mask_enc_1d = tokenize_and_distort_sentences(key, tokenizer, sentences, sequence_len)
    return src, mask_enc_1d, dst, mask_dec_1d

if __name__ == '__main__':
    set_start_method('spawn')

    # list files

    filenames = glob('./dump/*/*')
    key, *subkeys = rand.split(key, num=len(filenames))

    # process

    with Pool() as p:
        xs = p.starmap(pipeline, zip(subkeys, filenames))

    # unzip

    src = []
    mask_enc_1d = []
    dst = []
    mask_dec_1d = []

    for src_, mask_enc_1d_, dst_, mask_dec_1d_ in xs:
        src.append(src_)
        mask_enc_1d.append(mask_enc_1d_)
        dst.append(dst_)
        mask_dec_1d.append(mask_dec_1d_)

    src = np.vstack(src)
    mask_enc_1d = np.vstack(mask_enc_1d)
    dst = np.vstack(dst)
    mask_dec_1d = np.vstack(mask_dec_1d)

    # test

    print(src.shape)
    print(mask_enc_1d.shape)
    print(dst.shape)
    print(mask_dec_1d.shape)

    print(src.dtype)
    print(mask_enc_1d.dtype)
    print(dst.dtype)
    print(mask_dec_1d.dtype)

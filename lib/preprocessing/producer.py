# TODO: better name

import functools
from glob import glob
import jax.numpy as np
import jax.random as rand
import math
import multiprocessing
import numpy as onp
from os.path import expanduser
import random
import signal
import sys
from tqdm import tqdm
from transformers import BartTokenizer, PreTrainedTokenizer
from typing import Any, List, Literal, NoReturn, Tuple, Union

from .distort_sentence import distort_sentence
from ..dataset.dummy.load_dummy import load_dummy
from ..random.wrapper import key2seed, split_key

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def load_enwiki() -> List[str]:
    filenames = glob(expanduser('~/.bart-base-jax/enwiki/dump2/*/*'))
    if not filenames:
        raise ValueError('Cannot find the dataset in ~/.bart-base-jax/enwiki/dump2.')

    sentences = []
    for filename in tqdm(filenames):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                sentences.append(line.rstrip('\n'))
    print(f'INFO: Loaded {len(sentences)} sentences.')
    return sentences

def load_sentences(dataset=Union[Literal['enwiki'], Literal['dummy']]) -> List[str]:
    if dataset == 'enwiki':
        return load_enwiki()
    if dataset == 'dummy':
        return load_dummy()
    raise ValueError(f"Dataset should be either 'enwiki' or 'dummy', but got {dataset}")

def transform(tokenizer: PreTrainedTokenizer, sentences: List[str], key: rand.KeyArray) -> onp.ndarray:
    keys = split_key(key, num=len(sentences))
    distorted_sentences = [distort_sentence(sentence, key=key) for sentence, key in zip(sentences, keys)]

    x = tokenizer(sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)
    y = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)

    src = x.input_ids.astype(np.uint32)
    mask_enc_1d = x.attention_mask.astype(np.bool_)
    dst = y.input_ids.astype(np.uint32)
    mask_dec_1d = y.attention_mask.astype(np.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d

def transform_(tokenizer: PreTrainedTokenizer, sentences_key: Tuple[List[str], rand.KeyArray]) -> onp.ndarray:
    sentences, key = sentences_key
    return transform(tokenizer, sentences, key)

def producer(queue: multiprocessing.Queue, dataset: str, n_workers: int, batch_size: int, key: rand.KeyArray, should_shuffle: bool) -> NoReturn:
    ctx = multiprocessing.get_context('spawn')  # TODO: can we change this to fork?
    pool = ctx.Pool(processes=n_workers)
    signal.signal(signal.SIGTERM, lambda *_: pool.terminate() or sys.exit())

    sentences = load_sentences(dataset=dataset)
    n_sentences = len(sentences)
    n_chunks = math.ceil(n_sentences / batch_size)
    queue.put(n_chunks)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    while True:
        sentences_ = sentences[:]

        if should_shuffle:
            key, subkey = split_key(key)
            seed = key2seed(subkey)
            rng = random.Random(seed)
            rng.shuffle(sentences_)

        sentences_chunked = chunks(sentences_, chunk_size=batch_size)
        assert len(sentences_chunked) == n_chunks
        print(f'INFO: Successfully split into {n_chunks} chunks.')

        key, *subkeys = split_key(key, num=n_chunks+1)
        for tokenization_result in pool.imap(functools.partial(transform_, tokenizer), zip(sentences_chunked, subkeys)):
            queue.put(tokenization_result)

# TODO: better name

import functools
import math
import multiprocessing
import random
import signal
import sys
from transformers import BartTokenizer
from typing import Any, List, Literal, NoReturn, Union

from .tokenization_worker import tokenization_worker
from ..dataset.dummy.load_dummy import load_dummy
from ..dataset.enwiki.load_enwiki import load_enwiki
from ..random.wrapper import KeyArray, key2seed, split_key

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def load_sentences(dataset=Union[Literal['enwiki'], Literal['dummy']]) -> List[str]:
    return {
        'enwiki': load_enwiki,
        'dummy': load_dummy,
    }[dataset]()

def producer(queue: multiprocessing.Queue, dataset: str, n_workers: int, batch_size: int, key: KeyArray, should_shuffle: bool) -> NoReturn:
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
        for tokenization_result in pool.imap(functools.partial(tokenization_worker, tokenizer), zip(sentences_chunked, subkeys)):
            queue.put(tokenization_result)

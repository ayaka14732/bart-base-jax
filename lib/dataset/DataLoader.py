import functools
from glob import glob
import jax.numpy as np
import jax.random as rand
import math
import multiprocessing
import os
import random
import signal
import sys
from tqdm import tqdm
from transformers import BartTokenizer, PreTrainedTokenizer
from typing import Any, List, NoReturn, Optional, Tuple

from .distort_sentence import distort_sentence
from ..random.wrapper import key2seed, split_key

def load_sentences(max_files: Optional[int]=None):
    filenames = glob(os.path.expanduser('~/.cache/dump2/*/*'))
    if not filenames:
        raise ValueError('Cannot find the dataset in ~/.cache/dump2.')
    if max_files is not None:
        filenames = filenames[:max_files]

    sentences = []
    for filename in tqdm(filenames):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                sentences.append(line.rstrip('\n'))
    print(f'INFO: Loaded {len(sentences)} sentences.')
    return sentences

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def transform(tokenizer: PreTrainedTokenizer, sentences: List[str], key: rand.KeyArray) -> np.ndarray:
    keys = split_key(key, num=len(sentences))
    distorted_sentences = [distort_sentence(sentence, key=key) for sentence, key in zip(sentences, keys)]

    x = tokenizer(sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)
    y = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)

    src = x.input_ids.astype(np.uint32)
    mask_enc_1d = x.attention_mask.astype(np.bool_)
    dst = y.input_ids.astype(np.uint32)
    mask_dec_1d = y.attention_mask.astype(np.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d

def transform_(tokenizer: PreTrainedTokenizer, sentences_key: Tuple[List[str], rand.KeyArray]) -> np.ndarray:
    sentences, key = sentences_key
    return transform(tokenizer, sentences, key)

def producer(queue: multiprocessing.Queue, n_workers: int, max_files: int, chunk_size: int, key: rand.KeyArray, should_shuffle: bool) -> NoReturn:
    ctx = multiprocessing.get_context('spawn')  # TODO: can we change this to fork?
    pool = ctx.Pool(processes=n_workers)
    signal.signal(signal.SIGTERM, lambda *_: pool.terminate() or sys.exit())

    sentences = load_sentences(max_files=max_files)
    n_sentences = len(sentences)
    n_chunks = math.ceil(n_sentences / chunk_size)
    queue.put(n_chunks)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    while True:
        sentences_ = sentences[:]

        if should_shuffle:
            key, subkey = split_key(key)
            seed = key2seed(subkey)
            rng = random.Random(seed)
            rng.shuffle(sentences_)

        sentences_chunked = chunks(sentences_, chunk_size=chunk_size)
        assert len(sentences_chunked) == n_chunks
        print(f'INFO: Successfully split into {n_chunks} chunks.')

        key, *subkeys = split_key(key, num=n_chunks+1)
        for tokenization_result in pool.imap(functools.partial(transform_, tokenizer), zip(sentences_chunked, subkeys)):
            queue.put(tokenization_result)

def make_data(src: np.ndarray, mask_enc_1d: np.ndarray, dst: np.ndarray, mask_dec_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # TODO: better name

    # TODO: is this part correct?
    labels = dst

    batch_size, *_ = dst.shape

    bos_id = 2

    eoss = np.ones((batch_size, 1), dtype=np.uint32) * bos_id
    dst = np.hstack((eoss, dst[:, 1:]))

    trues = np.ones((batch_size, 1), dtype=np.bool_)
    mask_dec_1d = np.hstack((trues, mask_dec_1d[:, 1:]))
    # end todo

    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    return src, dst, mask_enc, mask_dec, mask_dec_enc, labels

class DataLoader:
    '''On-demand data loader.'''

    def __init__(self, key: rand.KeyArray, n_workers: Optional[int]=None, max_files: Optional[int]=None, queue_size: int=128, chunk_size: int=1024, should_shuffle=True) -> None:
        ctx = multiprocessing.get_context('spawn')

        queue = ctx.Queue(maxsize=queue_size)

        process = ctx.Process(target=producer, args=(queue, n_workers, max_files, chunk_size, key, should_shuffle))
        process.start()

        n_chunks = queue.get()

        self.n_chunks = n_chunks
        self.queue = queue
        self.process = process

    def __iter__(self):
        get = self.queue.get
        return (make_data(*get()) for _ in range(self.n_chunks))

    def close(self):
        os.kill(self.process.pid, signal.SIGTERM)
        self.process.terminate()
        print('Data loader process terminated...')

import functools
from glob import glob
import jax.numpy as np
import jax.random as rand
import multiprocessing
import os
import random
from tqdm import tqdm
from transformers import BartTokenizer, PreTrainedTokenizer
from typing import Any, List, Optional, Tuple

from .distort_sentence import distort_sentence
from ..random.wrapper import key2seed, split_key

def load_all_sentences(limit: Optional[int]=None):
    filenames = glob(os.path.expanduser('~/.cache/dump2/*/*'))
    if not filenames:
        raise ValueError('Cannot find the dataset in ~/.cache/dump2.')
    if limit is not None:
        filenames = filenames[:limit]

    all_sentences = []
    for filename in tqdm(filenames):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                all_sentences.append(line.rstrip('\n'))
    return all_sentences

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def transform(tokenizer: PreTrainedTokenizer, sentences: List[str], key: rand.KeyArray) -> np.ndarray:
    keys = split_key(key, num=len(sentences))
    distorted_sentences = [distort_sentence(sentence, key=key) for sentence, key in zip(sentences, keys)]

    x = tokenizer(sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True)
    y = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True)

    src = x.input_ids.astype(np.uint32)
    mask_enc_1d = x.attention_mask.astype(np.bool_)
    dst = y.input_ids.astype(np.uint32)
    mask_dec_1d = y.attention_mask.astype(np.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d

def transform_(tokenizer: PreTrainedTokenizer, sentences_key: Tuple[List[str], rand.KeyArray]) -> np.ndarray:
    sentences, key = sentences_key
    return transform(tokenizer, sentences, key)

def producer(queue: multiprocessing.Queue, n_epochs: int, n_workers: int, limit: int, chunk_size: int, key: rand.KeyArray):
    print('INFO: Loading sentences...')
    all_sentences = load_all_sentences(limit=limit)
    print(f'INFO: Loaded {len(all_sentences)} sentences.')

    key, subkey = split_key(key, num=2)
    seed = key2seed(subkey)
    rng = random.Random(seed)
    rng.shuffle(all_sentences)
    print('INFO: Sentence shuffling completed.')

    all_sentences_chunked = chunks(all_sentences, chunk_size=chunk_size)
    n_chunks = len(all_sentences_chunked)
    print(f'INFO: Successfully split into {n_chunks} chunks.')
    queue.put(n_chunks)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    ctx = multiprocessing.get_context('spawn')  # TODO: can we change this to fork?
    with ctx.Pool(processes=n_workers) as p:
        for _ in range(n_epochs):
            key, *subkeys = split_key(key, num=n_chunks+1)
            for tokenization_result in p.imap(functools.partial(transform_, tokenizer), zip(all_sentences_chunked, subkeys)):
                queue.put(tokenization_result)

def data_loader(key: rand.KeyArray, n_epochs: int, n_workers: Optional[int]=None, queue_size: int=256, limit: Optional[int]=None, chunk_size: int=1024):
    '''
    An on-demand data loader.

    Example:

    ```python
    from lib.dataset.data_loader import data_loader
    from lib.random.wrapper import seed2key

    if __name__ == '__main__':
        key = seed2key(seed=42)
        n_epochs = 4

        data_iter = data_loader(key=key, n_epochs=n_epochs, n_workers=8, limit=8)

        for _ in range(n_epochs):
            for i, (src, mask_enc_1d, dst, mask_dec_1d) in enumerate(data_iter()):
                print(i, tuple(map(lambda x: x.dtype, (src, mask_enc_1d, dst, mask_dec_1d))))
    ```
    '''
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue(maxsize=queue_size)
    producer_proc = ctx.Process(target=producer, args=(queue, n_epochs, n_workers, limit, chunk_size, key))
    producer_proc.start()

    n_chunks = queue.get()

    return lambda: (queue.get() for _ in range(n_chunks))

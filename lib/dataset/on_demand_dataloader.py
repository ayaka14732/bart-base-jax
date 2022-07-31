import functools
from glob import glob
import jax.numpy as np
import jax.random as rand
import multiprocessing
import os
import random
from tqdm import tqdm
from transformers import BartTokenizer, PreTrainedTokenizer
from typing import Any, List, Tuple

from .distort_sentence import distort_sentence
from ..random.wrapper import key2seed, split_key

def load_all_sentences():
    all_sentences = []
    for filename in tqdm(glob('/home/clara/bart-base-jax/dump2/*/*')[:128]):  #  TODO: avoid hardcoded path
        with open(filename, encoding='utf-8') as f:
            for line in f:
                all_sentences.append(line.rstrip('\n'))
    return all_sentences

def chunks(lst: List[Any], chunksize: int) -> List[List[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunksize] for i in range(0, len(lst), chunksize)]

def transform(tokenizer: PreTrainedTokenizer, sentences: List[str], key: rand.KeyArray) -> np.ndarray:
    keys = rand.split(key, num=len(sentences))
    distorted_sentences = [distort_sentence(sentence, key=key) for sentence, key in zip(sentences, keys)]

    x = tokenizer(sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True)
    y = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True)

    src = x.input_ids.astype(np.int32)
    mask_enc_1d = x.attention_mask
    dst = y.input_ids.astype(np.int32)
    mask_dec_1d = y.attention_mask

    return src, mask_enc_1d, dst, mask_dec_1d

def transform_(tokenizer: PreTrainedTokenizer, sentences_key: Tuple[List[str], rand.KeyArray]) -> np.ndarray:
    sentences, key = sentences_key
    return transform(tokenizer, sentences, key)

def producer(queue: multiprocessing.Queue, n_workers: int, key: rand.KeyArray):
    print('INFO: Loading sentences...')
    all_sentences = load_all_sentences()
    print(f'INFO: Loaded {len(all_sentences)} sentences.')

    key, subkey = split_key(key, num=2)
    seed = key2seed(subkey)
    rng = random.Random(seed)
    rng.shuffle(all_sentences)
    print('INFO: Sentence shuffling completed.')

    chunksize = 1024
    all_sentences_chunked = chunks(all_sentences, chunksize=chunksize)
    n_chunks = len(all_sentences_chunked)
    print(f'INFO: Successfully split into {n_chunks} chunks.')
    queue.put(n_chunks)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    keys = rand.split(key, num=n_chunks)
    with multiprocessing.get_context('spawn').Pool(processes=n_workers) as p:
        for tokenization_result in p.imap(functools.partial(transform_, tokenizer), zip(all_sentences_chunked, keys)):
            queue.put(tokenization_result)

def on_demand_dataloader(key: rand.KeyArray, n_workers: int=None):
    if n_workers is None:
        n_workers = os.cpu_count()

    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue(maxsize=n_workers)
    producer_proc = ctx.Process(target=producer, args=(queue, n_workers, key))
    producer_proc.start()

    length = queue.get()
    print(f'INFO: Started to load {length} chunks...')

    return (queue.get() for _ in range(length))

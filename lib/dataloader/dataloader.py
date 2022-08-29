from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as onp
import random
from typing import Any, List, Optional

from .device_split import device_split
from .tokenization_worker import tokenization_worker
from ..dataset.dummy.load_dummy import load_dummy
from ..dataset.enwiki.load_enwiki import load_enwiki
from ..random.wrapper import KeyArray, key2seed, split_key

Data = namedtuple('Data', (
    'src',
    'dst',
    'mask_enc_1d',
    'mask_dec_1d',
    'mask_enc',
    'mask_dec',
    'mask_dec_enc',
    'labels',
))

def make_data(src: onp.ndarray, mask_enc_1d: onp.ndarray, dst: onp.ndarray, mask_dec_1d: onp.ndarray) -> Data:
    # TODO: better name

    # TODO: is this part correct?
    labels = dst

    batch_size, *_ = dst.shape

    bos_id = 2

    eoss = onp.ones((batch_size, 1), dtype=onp.uint32) * bos_id
    dst = onp.hstack((eoss, dst[:, 1:]))

    trues = onp.ones((batch_size, 1), dtype=onp.bool_)
    mask_dec_1d = onp.hstack((trues, mask_dec_1d[:, 1:]))
    # end todo

    mask_enc = onp.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = onp.tril(onp.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = onp.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    # TODO: flexible batch size

    src = device_split(src)
    dst = device_split(dst)
    mask_dec_1d = device_split(mask_dec_1d)
    mask_enc = device_split(mask_enc)
    mask_dec = device_split(mask_dec)
    mask_dec_enc = device_split(mask_dec_enc)
    labels = device_split(labels)

    return Data(src, dst, mask_enc_1d, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels)

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def dataloader(dataset: str, key: KeyArray, batch_size: int, n_workers: Optional[int]=None, chunk_size: Optional[int]=1024, should_shuffle: bool=True):
    sentences = {
        'enwiki': load_enwiki,
        'dummy': load_dummy,
    }[dataset]()

    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        if should_shuffle:
            key, subkey = split_key(key)
            seed = key2seed(subkey)
            rng = random.Random(seed)
            rng.shuffle(sentences)

        sentences_chunked = chunks(sentences, chunk_size=chunk_size)
        n_sentences = len(sentences)
        n_chunks = len(sentences_chunked)
        print(f'INFO: Successfully split {n_sentences} sentences into {n_chunks} chunks.')

        subkeys = split_key(key, num=n_chunks); del key
        results = executor.map(tokenization_worker, zip(sentences_chunked, subkeys))

        src_ = None
        mask_enc_1d_ = None
        dst_ = None
        mask_dec_1d_ = None

        for src, mask_enc_1d, dst, mask_dec_1d in results:
            if src_ is not None:
                src = onp.vstack((src_, src))
                mask_enc_1d = onp.vstack((mask_enc_1d_, mask_enc_1d))
                dst = onp.vstack((dst_, dst))
                mask_dec_1d = onp.vstack((mask_dec_1d_, mask_dec_1d))

            while True:
                if src.shape[0] < batch_size:
                    src_ = src
                    mask_enc_1d_ = mask_enc_1d
                    dst_ = dst
                    mask_dec_1d_ = mask_dec_1d

                    break

                elif src.shape[0] == batch_size:
                    src_ = None
                    mask_enc_1d_ = None
                    dst_ = None
                    mask_dec_1d_ = None

                    yield make_data(src, mask_enc_1d, dst, mask_dec_1d)
                    break

                else:
                    src_ = None
                    mask_enc_1d_ = None
                    dst_ = None
                    mask_dec_1d_ = None

                    yield make_data(src[:batch_size], mask_enc_1d[:batch_size], dst[:batch_size], mask_dec_1d[:batch_size])

                    src = src[batch_size:]
                    mask_enc_1d = mask_enc_1d[batch_size:]
                    dst = dst[batch_size:]
                    mask_dec_1d = mask_dec_1d[batch_size:]

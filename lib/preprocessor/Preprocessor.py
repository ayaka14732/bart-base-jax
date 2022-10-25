import jax
import multiprocessing
import numpy as onp
import random
from typing import Any, Optional

from .prepare_data_for_model import prepare_data_for_model
from .tokenization_worker import tokenization_worker
from ..random.wrapper import KeyArray, key2seed, split_key
from ..utils import ProcessPoolExecutorWithQueueSizeLimit

def chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

class Preprocessor:
    def __init__(self, sentences: list[str], key: KeyArray, batch_size_per_device: int, n_workers: Optional[int]=None, queue_size: int=64, chunk_size: Optional[int]=2048, should_shuffle: bool=True):
        n_local_devices = jax.local_device_count()
        process_index = jax.process_index()
        process_count = jax.process_count()

        batch_size = batch_size_per_device * n_local_devices

        # TODO: is it plausible to split sentences at preprocessing time?
        n_sentences_per_device = len(sentences) // process_count
        sentences_sliced = sentences[process_index * n_sentences_per_device:(process_index + 1) * n_sentences_per_device]

        self.sentences = sentences_sliced
        self.key = key
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.queue_size = queue_size
        self.chunk_size = chunk_size
        self.should_shuffle = should_shuffle

    def __iter__(self):
        sentences = self.sentences

        if self.should_shuffle:
            self.key, subkey = split_key(self.key)
            seed = key2seed(subkey)
            rng = random.Random(seed)
            rng.shuffle(sentences)

        sentences_chunked = chunks(sentences, chunk_size=self.chunk_size)
        n_sentences = len(sentences)
        n_chunks = len(sentences_chunked)
        print(f'INFO: Successfully split {n_sentences} sentences into {n_chunks} chunks.')

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutorWithQueueSizeLimit(queue_size=self.queue_size, max_workers=self.n_workers, mp_context=ctx) as executor:
            self.key, *subkeys = split_key(self.key, num=n_chunks)
            results = executor.map(tokenization_worker, sentences_chunked, subkeys)

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
                    if src.shape[0] < self.batch_size:
                        src_ = src
                        mask_enc_1d_ = mask_enc_1d
                        dst_ = dst
                        mask_dec_1d_ = mask_dec_1d

                        break

                    elif src.shape[0] == self.batch_size:
                        src_ = None
                        mask_enc_1d_ = None
                        dst_ = None
                        mask_dec_1d_ = None

                        yield prepare_data_for_model(src, mask_enc_1d, dst, mask_dec_1d)
                        break

                    else:
                        src_ = None
                        mask_enc_1d_ = None
                        dst_ = None
                        mask_dec_1d_ = None

                        yield prepare_data_for_model(src[:self.batch_size], mask_enc_1d[:self.batch_size], dst[:self.batch_size], mask_dec_1d[:self.batch_size])

                        src = src[self.batch_size:]
                        mask_enc_1d = mask_enc_1d[self.batch_size:]
                        dst = dst[self.batch_size:]
                        mask_dec_1d = mask_dec_1d[self.batch_size:]

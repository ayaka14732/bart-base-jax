from collections import namedtuple
from jax._src.random import KeyArray
import multiprocessing
import numpy as onp
from os import kill
import signal
from typing import Optional

from .device_split import device_split
from .producer import producer

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

class DataLoader:
    '''On-demand data loader.'''

    def __init__(self, key: KeyArray, dataset: str, n_workers: Optional[int]=None, queue_size: int=128, batch_size: int=1024, should_shuffle=True) -> None:
        ctx = multiprocessing.get_context('spawn')

        queue = ctx.Queue(maxsize=queue_size)

        process = ctx.Process(target=producer, args=(queue, dataset, n_workers, batch_size, key, should_shuffle))
        process.start()

        n_chunks = queue.get()

        self.n_chunks = n_chunks
        self.queue = queue
        self.process = process

    def __iter__(self):
        get = self.queue.get
        return (make_data(*get()) for _ in range(self.n_chunks))

    def close(self):
        kill(self.process.pid, signal.SIGTERM)
        self.process.terminate()
        print('Data loader process terminated...')

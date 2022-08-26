from collections import namedtuple
import jax.numpy as np
import jax.random as rand
import multiprocessing
from os import kill
import signal
from typing import Optional, Tuple

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

def make_data(src: np.ndarray, mask_enc_1d: np.ndarray, dst: np.ndarray, mask_dec_1d: np.ndarray, n_devices: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # TODO: flexible batch size

    src = device_split(src, n_devices)
    dst = device_split(dst, n_devices)
    mask_dec_1d = device_split(mask_dec_1d, n_devices)
    mask_enc = device_split(mask_enc, n_devices)
    mask_dec = device_split(mask_dec, n_devices)
    mask_dec_enc = device_split(mask_dec_enc, n_devices)
    labels = device_split(labels, n_devices)

    return Data(src, dst, mask_enc_1d, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels)

class DataLoader:
    '''On-demand data loader.'''

    def __init__(self, key: rand.KeyArray, n_devices: int, n_workers: Optional[int]=None, max_files: Optional[int]=None, queue_size: int=128, batch_size: int=1024, should_shuffle=True) -> None:
        ctx = multiprocessing.get_context('spawn')

        queue = ctx.Queue(maxsize=queue_size)

        process = ctx.Process(target=producer, args=(queue, n_workers, max_files, batch_size, key, should_shuffle))
        process.start()

        n_chunks = queue.get()

        self.n_devices = n_devices
        self.n_chunks = n_chunks
        self.queue = queue
        self.process = process

    def __iter__(self):
        get = self.queue.get
        return (make_data(*get(), n_devices=self.n_devices) for _ in range(self.n_chunks))

    def close(self):
        kill(self.process.pid, signal.SIGTERM)
        self.process.terminate()
        print('Data loader process terminated...')

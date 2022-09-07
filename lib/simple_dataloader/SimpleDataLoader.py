from collections import namedtuple
import jax
import jax.numpy as np
import jax.random as rand
import math
import numpy as onp
from typing import Optional

from ..param_utils.load_params import load_params
from ..random.wrapper import KeyArray, split_key

pad_token_id = 0

device_default = jax.devices()[0]
device_cpu = jax.devices('cpu')[0]
put_default = lambda x: jax.device_put(x, device_default)
put_cpu = lambda x: jax.device_put(x, device_cpu)

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

class SimpleDataLoader:
    def __init__(self, path: str, batch_size: int, shuffle: bool=True, key: Optional[KeyArray]=None) -> None:
        if shuffle:
            assert key is not None

        dataset = load_params(path)  # dict of `onp.array`s

        self.src = dataset['src']
        self.mask_enc_1d = dataset['mask_enc_1d']
        self.dst = dataset['dst']
        self.mask_dec_1d = dataset['mask_dec_1d']

        self.key = key
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        src = put_cpu(self.src)
        mask_enc_1d = put_cpu(self.mask_enc_1d)
        dst = put_cpu(self.dst)
        mask_dec_1d = put_cpu(self.mask_dec_1d)

        key = self.key
        batch_size = self.batch_size

        dataset_len = len(src)

        idx = put_cpu(onp.arange(dataset_len))

        if self.shuffle:
            key, subkey = split_key(key)
            idx = rand.shuffle(subkey, idx)  # shuffled indices

            src = src[idx]
            mask_enc_1d = mask_enc_1d[idx]
            dst = dst[idx]
            mask_dec_1d = mask_dec_1d[idx]

        n_batches = math.floor(dataset_len / batch_size)

        for i in range(n_batches):
            src_ = src[i * batch_size:(i + 1) * batch_size]
            mask_enc_1d_ = mask_enc_1d[i * batch_size:(i + 1) * batch_size]
            dst_ = dst[i * batch_size:(i + 1) * batch_size]
            mask_dec_1d_ = mask_dec_1d[i * batch_size:(i + 1) * batch_size]

            mask_enc = np.einsum('bi,bj->bij', mask_enc_1d_, mask_enc_1d_)[:, None]
            mask_dec = np.tril(onp.einsum('bi,bj->bij', mask_dec_1d_, mask_dec_1d_))[:, None]
            mask_dec_enc = np.einsum('bi,bj->bij', mask_dec_1d_, mask_enc_1d_)[:, None]

            pad_at_end = put_cpu(onp.ones((batch_size, 1), dtype=onp.uint16) * pad_token_id)
            labels = np.hstack((dst_[:, 1:], pad_at_end))

            data = src_, dst_, mask_enc_1d_, mask_dec_1d_, mask_enc, mask_dec, mask_dec_enc, labels
            data = Data(*map(put_default, data))
            yield data

        self.key = key

import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import jax

from lib.preprocessing.DataLoader import DataLoader
from lib.random.wrapper import seed2key

jax.config.update('jax_platforms', 'cpu')

if __name__ == '__main__':
    key = seed2key(42)

    data_loader = DataLoader(key=key, dataset='dummy', n_workers=24)
    for n_batches, batch in enumerate(data_loader):
        print(
            batch.src.shape,
            batch.dst.shape,
            batch.mask_enc_1d.shape,
            batch.mask_dec_1d.shape,
            batch.mask_enc.shape,
            batch.mask_dec.shape,
            batch.mask_dec_enc.shape,
            batch.labels.shape,
        )
    data_loader.close()

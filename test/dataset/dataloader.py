import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import jax

from lib.dataloader.dataloader import dataloader
from lib.random.wrapper import seed2key

jax.config.update('jax_platforms', 'cpu')

if __name__ == '__main__':
    key = seed2key(42)

    data_loader = dataloader(dataset='dummy', batch_size=48, key=key, n_workers=32)
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

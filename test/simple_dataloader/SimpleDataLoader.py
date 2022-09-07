import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.random.wrapper import seed2key
from lib.simple_dataloader.SimpleDataLoader import SimpleDataLoader

key = seed2key(42)

data_loader = SimpleDataLoader(key, 'dataset.dat', batch_size=22)

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
    print(batch.src.device())

import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as np
import optax
import time
import wandb

from lib.model.fwd_transformer import fwd_transformer
from lib.param_utils.load_params import load_params
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key, split_key
from lib.simple_dataloader.SimpleDataLoader import SimpleDataLoader
from lib.training.cross_entropy_loss import cross_entropy_loss

vocab_size = 6995
pad_token_id = 0
optimizer = None

device_default = jax.devices()[0]
device_cpu = jax.devices('cpu')[0]
put_default = lambda x: jax.device_put(x, device_default)
put_cpu = lambda x: jax.device_put(x, device_cpu)

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['lm_head']
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d, n_classes=vocab_size) / len(labels)
    return loss

@jax.jit
def train_step(params, opt_state, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

def main():
    n_epochs = 25
    batch_size = 35
    learning_rate = 0.03
    freeze_lr_scale = 0.25

    wandb.init(project='bart-finetune-twblg', config={
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'freeze_lr_scale': freeze_lr_scale,
        'untie_lm_head': True,
    })

    key = seed2key(seed=42)

    key, subkey = split_key(key)
    data_loader = SimpleDataLoader('train.dat', batch_size, key=subkey)

    params = load_params('untrained_params.dat')
    params = put_default(params)

    global optimizer
    param_labels = {
        'embedding': 'freeze',
        'encoder_embed_positions': 'freeze',
        'encoder_embed_layer_norm': 'freeze',
        'encoder_layers': 'freeze',
        'decoder_embed_positions': 'train',
        'decoder_embed_layer_norm': 'train',
        'decoder_layers': 'train',
        'lm_head': 'train',
    }
    optimizer_scheme = {
        'train': optax.chain(
            optax.adaptive_grad_clip(0.1, eps=0.001),
            optax.sgd(learning_rate=learning_rate),
        ),
        'freeze': optax.chain(
            optax.adaptive_grad_clip(0.1, eps=0.001),
            optax.sgd(learning_rate=learning_rate * freeze_lr_scale),
        ),
    }
    optimizer = optax.multi_transform(optimizer_scheme, param_labels)
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        epoch_loss = 0.

        for n_batches, batch in enumerate(data_loader):
            start_time = time.time()

            key, subkey = split_key(key)
            subkey = put_default(subkey)
            params, opt_state, loss = train_step(
                params,
                opt_state,
                batch.src,
                batch.dst,
                batch.mask_dec_1d,
                batch.mask_enc,
                batch.mask_dec,
                batch.mask_dec_enc,
                batch.labels,
                dropout_key=subkey,
            )

            batch_loss = loss.item()
            assert not np.isnan(batch_loss)
            epoch_loss += batch_loss

            elapsed_time = time.time() - start_time

            wandb.log({'batch loss': batch_loss, 'time': elapsed_time})

        epoch_loss /= n_batches
        wandb.log({'epoch': epoch, 'epoch loss': epoch_loss})

        filename = f'{wandb.run.name}.dat'
        save_params(params, filename)

if __name__ == '__main__':
    main()

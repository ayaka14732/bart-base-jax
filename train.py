import functools
import jax
import jax.numpy as np
import numpy as onp
import optax
import time
import wandb

from lib.dataset.data_loader import data_loader
from lib.model.fwd_transformer import fwd_transformer
from lib.param_utils.init_params import init_params
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss

# hyperparameters

devices = jax.devices()
n_devices = jax.device_count()
assert n_devices == 8

n_epochs = 1
batch_size = 16 * n_devices  # 28 * n_devices
learning_rate = 0.023

def get_tokenizer_info():
    from transformers import BartTokenizer

    en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    vocab_size = en_tokenizer.vocab_size
    pad_token_id = en_tokenizer.pad_token_id

    return vocab_size, pad_token_id

vocab_size, pad_token_id = get_tokenizer_info()

wandb.init(project='bart-pretraining', config={
    'n_devices': n_devices,
    'n_epochs': n_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'vocab_size': vocab_size,
    'pad_token_id': pad_token_id,
})


optimizer = optax.adam(learning_rate=learning_rate)

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask=mask_dec, num_classes=vocab_size) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='num_devices')
def train_step(params, opt_state, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = train_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)

    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

def device_split(arr):
    '''Splits the first axis of `arr` evenly across the number of devices.'''
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

def mask_1d_to_2d(mask_enc_1d, mask_dec_1d):
    mask_enc = device_split(np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None])
    mask_dec = device_split(np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None])
    mask_dec_enc = device_split(np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None])
    return mask_enc, mask_dec, mask_dec_enc

def save_checkpoint(replicated_params) -> None:
    params = jax.tree_map(lambda x: x[0], replicated_params)
    filename = f'{wandb.run.name}.dat'
    save_params(params, filename)

# TODO:
# `data_loader` should yield
# src, dst, mask_enc, mask_dec, mask_dec_enc, labels
#
# should be able to close the child processes (early stopping)

def main():
    params = init_params()

    opt_state = optimizer.init(params)

    key = seed2key(seed=42)

    data_iter = data_loader(key=key, n_epochs=n_epochs, n_workers=8, limit=8)

    replicated_params = jax.device_put_replicated(params, devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, devices)

    for _ in range(n_epochs):

        epoch_loss = 0.

        for i, (src, mask_enc_1d, dst, mask_dec_1d) in enumerate(data_iter()):
            start_time = time.time()

            labels = onp.hstack((dst[:, 1:], np.ones((len(batch), 1), dtype=np.int32) * pad_token_id))

            src = device_split(src)
            dst = device_split(dst)
            labels = device_split(labels)

            batch_mask_enc_1d = mask_enc_1d[batch]
            batch_mask_dec_1d = mask_dec_1d[batch]

            mask_enc = np.einsum('bi,bj->bij', batch_mask_enc_1d, batch_mask_enc_1d)[:, None]
            mask_dec = np.tril(np.einsum('bi,bj->bij', batch_mask_dec_1d, batch_mask_dec_1d))[:, None]
            mask_dec_enc = np.einsum('bi,bj->bij', batch_mask_dec_1d, batch_mask_enc_1d)[:, None]

            mask_enc = device_split(mask_enc)
            mask_dec = device_split(mask_dec)
            mask_dec_enc = device_split(mask_dec_enc)

            key, subkey = split_key(key, nums=9)

            replicated_params, replicated_opt_state, replicated_loss = train_step(replicated_params, replicated_opt_state, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=subkey)

            batch_loss = replicated_loss[0].item()
            assert not np.isnan(batch_loss)
            epoch_loss += batch_loss

            elapsed_time = time.time() - start_time

            wandb.log({'batch loss': batch_loss, 'time': elapsed_time})

        epoch_loss /= n_batches
        wandb.log({'epoch loss': epoch_loss})

        save_checkpoint(replicated_params)

if __name__ == '__main__':
    main()

import functools
import jax
import jax.numpy as np
import optax
import time
import wandb

from lib.dataset.DataLoader import DataLoader
from lib.model.fwd_transformer import fwd_transformer
from lib.param_utils.init_params import init_params
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss

from lib.training_temp import device_split

vocab_size = 50265  # BartTokenizer.from_pretrained('facebook/bart-base').vocab_size
pad_token_id = 1  # BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id
global_vars = {
    'optimizer': None,
}

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

    updates, opt_state = global_vars['optimizer'].update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

def main():
    # hyperparameters

    devices = jax.devices()
    n_devices = jax.device_count()
    assert n_devices == 8

    n_epochs = 2
    batch_size = 16 * n_devices  # 28 * n_devices
    learning_rate = 0.023

    wandb.init(project='bart-pretraining', config={
        'n_devices': n_devices,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    })

    key = seed2key(seed=42)

    key, subkey = split_key(key)
    data_loader = DataLoader(key=subkey, n_workers=12, max_files=8)

    key, subkey = split_key(key)
    params = init_params(key=subkey)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    global_vars['optimizer'] = optimizer

    replicated_params = jax.device_put_replicated(params, devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, devices)

    for _ in range(n_epochs):
        epoch_loss = 0.

        for n_batches, (src, dst, mask_enc, mask_dec, mask_dec_enc, labels) in enumerate(data_loader):
            start_time = time.time()

            src = device_split(src, n_devices)
            dst = device_split(dst, n_devices)
            mask_enc = device_split(mask_enc, n_devices)
            mask_dec = device_split(mask_dec, n_devices)
            mask_dec_enc = device_split(mask_dec_enc, n_devices)
            labels = device_split(labels, n_devices)

            key, subkey = split_key(key); subkeys = split_key(subkey, num=n_devices)
            replicated_params, replicated_opt_state, replicated_loss = train_step(replicated_params, replicated_opt_state, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=subkeys)

            batch_loss = replicated_loss[0].item()
            assert not np.isnan(batch_loss)
            epoch_loss += batch_loss

            elapsed_time = time.time() - start_time

            wandb.log({'batch loss': batch_loss, 'time': elapsed_time})

        epoch_loss /= n_batches
        wandb.log({'epoch loss': epoch_loss})

        params = jax.tree_map(lambda x: x[0], replicated_params)
        filename = f'{wandb.run.name}.dat'
        save_params(params, filename)

    data_loader.close()

if __name__ == '__main__':
    main()

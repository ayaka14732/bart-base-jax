import functools
import jax
import jax.numpy as np
import optax
import time

from lib.dataloader.DataLoader import Data, DataLoader
from lib.model.fwd_transformer import fwd_transformer
from lib.param_utils.init_params import init_params
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss

jax.distributed.initialize()

process_index = jax.process_index()

if process_index == 0: 
    import wandb

pad_token_id = 1  # BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id
optimizer = None

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='n_devices')
def train_step(params, opt_state, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)

    grads = jax.lax.pmean(grads, axis_name='n_devices')
    loss = jax.lax.pmean(loss, axis_name='n_devices')

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

def main():
    # hyperparameters

    devices = jax.devices()
    n_devices = jax.device_count()
    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()
    print('Number of devices:', n_devices)

    n_epochs = 2
    batch_size = 22 * n_devices  # 28 * n_devices
    learning_rate = 0.023

    if process_index == 0:
        wandb.init(project='bart-pretraining', config={
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        })

    key = seed2key(seed=42 + process_index)

    key, subkey = split_key(key)
    data_loader = DataLoader(dataset='enwiki', key=subkey, batch_size=batch_size, n_workers=50)

    key, subkey = split_key(key)
    params = init_params(key=subkey)

    global optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    replicated_params = jax.device_put_replicated(params, local_devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, local_devices)

    for _ in range(n_epochs):
        if process_index == 0:
            epoch_loss = 0.

        for n_batches, batch in enumerate(data_loader):
            if process_index == 0:
                start_time = time.time()


            batch = Data(*map(lambda a: a[process_index * 4:(process_index + 1) * 4], batch))

            key, subkey = split_key(key); subkeys = split_key(subkey, num=n_local_devices)  # force `subkeys` to be an array instead of a list
            replicated_params, replicated_opt_state, replicated_loss = train_step(
                replicated_params,
                replicated_opt_state,
                batch.src,
                batch.dst,
                batch.mask_dec_1d,
                batch.mask_enc,
                batch.mask_dec,
                batch.mask_dec_enc,
                batch.labels,
                dropout_key=subkeys,
            )

            if process_index == 0:
                batch_loss = replicated_loss[0].item()
                assert not np.isnan(batch_loss)
                epoch_loss += batch_loss

                elapsed_time = time.time() - start_time

                wandb.log({'batch loss': batch_loss, 'time': elapsed_time})

        if process_index == 0:
            epoch_loss /= n_batches
            wandb.log({'epoch loss': epoch_loss})

            params = jax.tree_map(lambda x: x[0], replicated_params)
            filename = f'{wandb.run.name}.dat'
            save_params(params, filename)

if __name__ == '__main__':
    main()

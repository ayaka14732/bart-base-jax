# from lib.init_chip import init_four_chip; init_four_chip()
import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import functools
import jax_smi
import optax
import os
import time
from transformers import FlaxMarianMTModel
import wandb

from lib.dataset.load_cantonese import load_cantonese
from lib.param_utils.save_params import save_params
from lib.preprocessor.Preprocessor import Preprocessor
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss

forward_inner = None
optimizer = None

def forward(params, src, dst, mask_enc_1d, mask_dec_1d, labels, dropout_key=None):
    outputs = forward_inner(
        input_ids=src,
        attention_mask=mask_enc_1d,
        decoder_input_ids=dst,
        decoder_attention_mask=mask_dec_1d,
        params=params,
        dropout_rng=dropout_key,
    )
    logits = outputs.logits
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d)
    return loss

@functools.partial(jax.pmap, axis_name='n_devices')
def train_tick(params, opt_state, src, dst, mask_enc_1d, mask_dec_1d, labels, dropout_key):
    loss, grads = jax.value_and_grad(forward)(params, src, dst, mask_enc_1d, mask_dec_1d, labels, dropout_key=dropout_key)

    grads = jax.lax.pmean(grads, axis_name='n_devices')
    loss = jax.lax.pmean(loss, axis_name='n_devices')

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

@functools.partial(jax.pmap, axis_name='n_devices')
def eval_tick(params, src, dst, mask_enc_1d, mask_dec_1d, labels):
    loss = forward(params, src, dst, mask_enc_1d, mask_dec_1d, labels)
    loss = jax.lax.pmean(loss, axis_name='n_devices')
    return loss

def main():
    # initialisation

    jax.distributed.initialize()
    jax_smi.initialise_tracking()
    jax.devices()  # force TPU initialisation
    jax.config.update('jax_platforms', 'cpu')  # suppress TPU in subprocesses
    process_index = jax.process_index()
    if process_index == 0:
        wandb.init(project='en-yue-baseline')

    # hyperparameters

    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()

    n_epochs = 10

    batch_size_per_device_train = 4
    batch_size_per_device_dev = 40

    key = seed2key(seed=42 + process_index)

    sentences_train = load_cantonese(split='train')
    sentences_dev = load_cantonese(split='dev')

    key, subkey = split_key(key)
    preprocessor_train = Preprocessor(sentences_train, key=subkey, batch_size_per_device=batch_size_per_device_train, n_workers=16)

    key, subkey = split_key(key)
    preprocessor_eval = Preprocessor(sentences_dev, key=subkey, batch_size_per_device=batch_size_per_device_dev, n_workers=16)

    model = FlaxMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    global forward_inner
    forward_inner = model.__call__
    params = model.params

    # remove model params
    cpu_device = jax.devices('cpu')[0]
    model.params = jax.tree_map(lambda a: jax.device_put(a, cpu_device), model.params)
    model.params

    global optimizer
    optimizer = optax.adamw(learning_rate=8e-6)
    opt_state = optimizer.init(params)

    replicated_params = jax.device_put_replicated(params, local_devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, local_devices)

    tick_total = 0

    for epoch in range(n_epochs):
        # train

        if process_index == 0:
            total_loss_train = 0.

        for tick_train, batch_train in enumerate(preprocessor_train, 1):
            if process_index == 0:
                start_time = time.time()

            key, subkey = split_key(key); subkeys = split_key(subkey, num=n_local_devices)  # force `subkeys` to be an array instead of a list
            replicated_params, replicated_opt_state, replicated_batch_loss_train = train_tick(
                replicated_params,
                replicated_opt_state,
                batch_train.src,
                batch_train.dst,
                batch_train.mask_enc_1d,
                batch_train.mask_dec_1d,
                batch_train.labels,
                dropout_key=subkeys,
            )

            if process_index == 0:
                # record loss and time
                batch_loss_train = replicated_batch_loss_train[0].item()
                total_loss_train += batch_loss_train
                elapsed_time = time.time() - start_time
                wandb.log({'train loss': batch_loss_train, 'time': elapsed_time}, commit=False)

            tick_total += 1

            if process_index == 0:
                wandb.log({'tick': tick_total}, commit=True)

        if process_index == 0:
            wandb.log({'epoch loss': total_loss_train / tick_train}, commit=False)

            # save params
            params = jax.tree_map(lambda x: x[0], replicated_params)
            filename = f'{wandb.run.name}-{epoch}.dat'
            save_params(params, filename + '.tmp')
            os.rename(filename + '.tmp', filename)

        del batch_train

        # eval

        if process_index == 0:
            total_loss_eval = 0.

        for tick_eval, batch_eval in enumerate(preprocessor_eval, 1):
            replicated_batch_loss_eval = eval_tick(
                replicated_params,
                batch_eval.src,
                batch_eval.dst,
                batch_eval.mask_enc_1d,
                batch_eval.mask_dec_1d,
                batch_eval.labels,
            )
            if process_index == 0:
                batch_loss_eval = replicated_batch_loss_eval[0].item()
                total_loss_eval += batch_loss_eval

        if process_index == 0:
            wandb.log({'eval loss': total_loss_eval / tick_eval, 'epoch': epoch}, commit=True)

        del batch_eval

if __name__ == '__main__':
    main()

import os; os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as np

import functools
import jax_smi
import optax
import time
import wandb

from lib.model import fwd_transformer_merged
from lib.param_utils.load_params import load_params
from lib.param_utils.save_params import save_params
from lib.preprocessor.Preprocessor import Preprocessor
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss
from lib.en_kfw_nmt.load_sentences import load_sentences


pad_token_id = 1  # BartTokenizerWithoutOverflowEOS.from_pretrained('facebook/bart-base').pad_token_id
optimizer = None

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_transformer_merged(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['lm_head']
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d)
    return loss

@jax.jit
def train_step(params, opt_state, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

@jax.jit
def eval_step(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_transformer_merged(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
    lm_head = params['lm_head']
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d)
    return loss

def main():
    # initialisation

    # jax.distributed.initialize()
    # jax_smi.initialise_tracking()
    jax.config.update('jax_platforms', 'cpu')  # suppress TPU in subprocesses
    process_index = jax.process_index()
    if process_index == 0:
        wandb.init(project='en-kfw-nmt')

    # hyperparameters

    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()

    n_epochs = 12

    batch_size_per_device_train = 24
    batch_size_per_device_dev = 24

    eval_every_n_steps = 1024
    save_every_n_steps = 20480

    key = seed2key(seed=42 + process_index)

    sentences_train = load_sentences(split='train')
    sentences_dev = load_sentences(split='dev')

    key, subkey = split_key(key)
    preprocessor_train = Preprocessor(sentences_train, key=subkey, batch_size_per_device=batch_size_per_device_train, n_workers=3)

    key, subkey = split_key(key)
    preprocessor_eval = Preprocessor(sentences_dev, key=subkey, batch_size_per_device=batch_size_per_device_dev, n_workers=3)

    key, subkey = split_key(key)
    params = load_params('params_merged.dat')
    params = jax.tree_map(np.asarray, params)

    learning_rate = 0.03
    param_labels = {
        'encoder_embedding': 'freeze',
        'encoder_embed_positions': 'freeze',
        'encoder_embed_layer_norm': 'freeze',
        'encoder_layers': ['freeze'] * 2 + ['train'] * 8 + ['freeze'] * 2,
        'decoder_embedding': 'freeze',
        'decoder_embed_positions': 'freeze',
        'decoder_embed_layer_norm': 'freeze',
        'decoder_layers': 'freeze',
        'lm_head': 'freeze',
    }
    optimizer_scheme = {
        'train': optax.chain(
            optax.adaptive_grad_clip(0.1, eps=0.001),
            optax.sgd(learning_rate=learning_rate),
        ),
        'freeze': optax.chain(
            optax.adaptive_grad_clip(0.1, eps=0.001),
            optax.sgd(learning_rate=learning_rate * 0.1),
        ),
    }

    global optimizer
    optimizer = optax.multi_transform(optimizer_scheme, param_labels)
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        if process_index == 0:
            epoch_loss_train = 0.

        for step, batch_train in enumerate(preprocessor_train):
            if process_index == 0:
                start_time = time.time()

            key, subkey = split_key(key)
            params, opt_state, batch_loss_train = train_step(
                params,
                opt_state,
                batch_train.src,
                batch_train.dst,
                batch_train.mask_dec_1d,
                batch_train.mask_enc,
                batch_train.mask_dec,
                batch_train.mask_dec_enc,
                batch_train.labels,
                dropout_key=subkey,
            )

            if process_index == 0:
                # record loss and time
                batch_loss_train = batch_loss_train.item()
                epoch_loss_train += batch_loss_train
                elapsed_time = time.time() - start_time
                wandb.log({'train loss': batch_loss_train, 'time': elapsed_time}, commit=False)

            # eval
            if step % eval_every_n_steps == 0:
                if process_index == 0:
                    total_loss_eval = 0.

                for batch_eval in preprocessor_eval:
                    batch_loss_eval = eval_step(
                        params,
                        batch_eval.src,
                        batch_eval.dst,
                        batch_eval.mask_dec_1d,
                        batch_eval.mask_enc,
                        batch_eval.mask_dec,
                        batch_eval.mask_dec_enc,
                        batch_eval.labels,
                    )
                    if process_index == 0:
                        batch_loss_eval = batch_loss_eval.item()
                        total_loss_eval += batch_loss_eval

                if process_index == 0:
                    wandb.log({'eval loss': total_loss_eval}, commit=False)

            if process_index == 0:
                wandb.log({}, commit=True)

        if process_index == 0:
            epoch_loss_train /= step
            wandb.log({'epoch loss': epoch_loss_train}, commit=False)

            # save params
            filename = f'{wandb.run.name}-{epoch}.dat'
            save_params(params, filename)

if __name__ == '__main__':
    main()

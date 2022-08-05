import functools
import jax
import jax.numpy as np
import jax.random as rand
import numpy as onp
import optax
import time
from transformers import BartTokenizer
import wandb

from lib import fwd_transformer
# from lib.param_utils.load_params import load_params
from lib.param_utils.init_params import init_params

# Procedure:
# 1. load a pretrained BART-base-chinese encoder
# 2. adding a linear layer to 1, substitute the embedding part of pretrained BART-base-english
# 3. fine-tune params including linear, first layer attention
# 4. fine-tune all params with decayed lr

devices = jax.local_devices()
n_devices = jax.local_device_count()
assert n_devices == 8

n_epoch = 1
batch_size = 16 * n_devices  # 28 * n_devices
learning_rate = 0.023

en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

wandb.init(project='bart-pretraining', config={
    'n_epoch': n_epoch,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'extra_description': 'using sgd optimizer; trained on wikimatrix21; 13th revision of freezing',
})

def cross_entropy_loss(logits, labels, mask):
    labels_onehot = jax.nn.one_hot(labels, num_classes=en_tokenizer.vocab_size)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    loss *= mask
    return np.sum(loss)

# 1. load params

params = init_params()

optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask=mask_dec) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='num_devices')
def train_step(params, opt_state, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = train_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)

    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

@jax.jit
def eval_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask=mask_dec) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='num_devices')
def eval_step(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels):
    loss = eval_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels)
    loss = jax.lax.pmean(loss, axis_name='num_devices')
    return loss

def device_split(arr):
    '''Splits the first axis of `arr` evenly across the number of devices.'''
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

def do_on_cpu(f):
    return jax.jit(f, backend='cpu')

def mask_1d_to_2d(mask_enc_1d, mask_dec_1d):
    mask_enc = device_split(np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None])
    mask_dec = device_split(np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None])
    mask_dec_enc = device_split(np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None])
    return mask_enc, mask_dec, mask_dec_enc

replicated_params = jax.device_put_replicated(params, devices)
replicated_opt_state = jax.device_put_replicated(opt_state, devices)

def save_ckpt():
    params = jax.tree_map(lambda x: x[0], replicated_params)
    filename = f'{wandb.run.name}.dat'
    save_params(params, filename)

key = rand.PRNGKey(42)
n_sents = len(input_ids)

permute = do_on_cpu(lambda key: rand.permutation(key, n_sents))

# jax.profiler.start_trace(log_dir='/tmp/jax-profiler')

for _ in range(1, n_epoch + 1):
    epoch_loss = 0.

    n_batches = n_sents // batch_size
    key, subkey = rand.split(key)
    shuffled_indices = onp.asarray(permute(subkey))

    for i in range(n_batches):
        start_time = time.time()

        batch = shuffled_indices[i*batch_size:(i+1)*batch_size]

        src = input_ids[batch]
        dst = decoder_input_ids[batch]
        labels = onp.hstack((dst[:, 1:], np.ones((len(batch), 1), dtype=np.int32) * en_tokenizer.pad_token_id))

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

        key, subkey = (lambda keys: (keys[0], keys[1:]))(rand.split(key, num=9))

        replicated_params, replicated_opt_state, replicated_loss = train_step(replicated_params, replicated_opt_state, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=subkey)

        batch_loss = replicated_loss[0].item()
        assert not np.isnan(batch_loss)
        epoch_loss += batch_loss

        elapsed_time = time.time() - start_time

        wandb.log({'batch loss': batch_loss, 'time': elapsed_time})

    epoch_loss /= n_batches
    wandb.log({'epoch loss': epoch_loss})

    save_ckpt()

# jax.profiler.stop_trace()

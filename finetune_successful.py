import functools
import jax
import jax.numpy as np
import jax.random as rand
import math
import numpy as onp
import optax
from transformers import BartTokenizer
from tqdm import trange
import wandb

from lib.load_dataset import load_dataset
from lib.param_utils.load_params import load_params
from lib.param_utils.save_params import save_params
from lib.fwd_nmt_transformer import fwd_nmt_transformer

# Procedure:
# 1. load a pretrained BART-base-chinese encoder
# 2. adding a linear layer to 1, substitute the embedding part of pretrained BART-base-english
# 3. fine-tune params including linear, first layer attention
# 4. fine-tune all params with decayed lr

devices = jax.local_devices()
n_devices = jax.local_device_count()
assert n_devices == 8

n_epoch = 1
batch_size = 18 * n_devices
learning_rate = 0.023

wandb.init(project='bart-nmt-zh-en', config={
    'n_epoch': n_epoch,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'extra_description': 'using sgd optimizer; trained on wikimatrix21; 13th revision of freezing',
})

jax.profiler.start_trace(log_dir='/tmp/jax-profiler')

def cross_entropy_loss(logits, labels, mask):
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    exp_loss = np.take_along_axis(softmax_probs, labels[..., None], axis=-1)
    loss = -np.log(exp_loss)
    loss = loss * mask
    return np.sum(loss)

# 1. load params

en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def make_params():
    params_ch = jax.tree_map(np.asarray, load_params('params_bart_base_zh.dat'))
    params_en = jax.tree_map(np.asarray, load_params('params_bart_base_en.dat'))
    return {
        'ch': {
            'embedding': params_ch['embedding'],
            'encoder_embed_positions': params_ch['encoder_embed_positions'],
            'encoder_embed_layer_norm': params_ch['encoder_embed_layer_norm'],
            'encoder_layers': params_ch['encoder_layers'],
        },
        'embedding': params_en['embedding'],
        'decoder_embed_positions': params_en['decoder_embed_positions'],
        'decoder_embed_layer_norm': params_en['decoder_embed_layer_norm'],
        'encoder_layers': params_en['encoder_layers'],
        'decoder_layers': params_en['decoder_layers'],
    }

params = make_params()

param_labels = {
    'ch': {
        'embedding': 'freeze',
        'encoder_embed_positions': 'train',
        'encoder_embed_layer_norm': 'train',
        'encoder_layers': 'train',
    },
    'encoder_layers': 'freeze',
    'embedding': 'freeze',
    'decoder_embed_positions': 'freeze',
    'decoder_embed_layer_norm': 'freeze',
    'decoder_layers': ['train', 'freeze', 'freeze', 'freeze', 'freeze', 'freeze'],
}

input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d = load_dataset('wikimatrix21.zh', 'wikimatrix21.en')
eval_input_ids, eval_mask_enc_1d, eval_decoder_input_ids, eval_mask_decoder_1d = load_dataset('dev/newsdev2017.zh', 'dev/newsdev2017.en')

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

optimizer = optax.multi_transform(optimizer_scheme, param_labels)
opt_state = optimizer.init(params)

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_nmt_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
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
    outputs = fwd_nmt_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
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

def evaluate(replicated_params):
    n_batches = len(eval_input_ids) // batch_size
    epoch_loss = 0.
    for i in range(n_batches):
        src = device_split(input_ids[i * batch_size:(i + 1) * batch_size])
        dst = device_split(decoder_input_ids[i * batch_size:(i + 1) * batch_size])
        labels = device_split(onp.hstack(
            (decoder_input_ids[i * batch_size:(i + 1) * batch_size, 1:],
             np.ones((batch_size, 1), dtype=np.int32) * en_tokenizer.pad_token_id)))
        mask_enc, mask_dec, mask_dec_enc = mask_1d_to_2d(mask_enc_1d[i * batch_size:(i + 1) * batch_size],
                                                         mask_dec_1d[i * batch_size:(i + 1) * batch_size])
        loss = eval_step(replicated_params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels)
        batch_loss = jax.device_get(jax.tree_map(lambda x: x[0], loss)).item()
        epoch_loss += batch_loss
    epoch_loss /= n_batches
    return epoch_loss

replicated_params = jax.device_put_replicated(params, devices)
replicated_opt_state = jax.device_put_replicated(opt_state, devices)

def save_ckpt():
    params = jax.tree_map(lambda x: x[0], replicated_params)
    filename = f'{wandb.run.name}.dat'
    save_params(params, filename)

key = rand.PRNGKey(42)
n_sents = len(input_ids)

for _ in trange(1, n_epoch + 1, desc='Epoch', smoothing=1.):
    epoch_loss = 0.
    eval_loss = math.inf

    n_batches = n_sents // batch_size
    key, subkey = rand.split(key)
    shuffled_indices = onp.asarray(do_on_cpu(lambda: rand.permutation(subkey, n_sents))())

    for i in trange(n_batches, desc='Batch', leave=False, smoothing=1.):
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
        wandb.log({'batch loss': batch_loss})
        epoch_loss += batch_loss

        if i % 2000 == 1999:
            new_eval_loss = evaluate(replicated_params)
            if new_eval_loss > eval_loss:
                break

            eval_loss = new_eval_loss
            save_ckpt()

    epoch_loss /= n_batches
    wandb.log({'epoch loss': epoch_loss})

    save_ckpt()

jax.profiler.stop_trace()

import jax
import jax.numpy as np
import jax.random as rand
import math
import numpy as onp
from transformers import BertTokenizer, BartTokenizer
from tqdm import trange
import optax
import functools
from dataloader import process_one_dataset

from lib.param_utils.load_params import load_params
from lib.fwd_nmt_transformer import fwd_nmt_transformer

#Procedure:
#1. load a pretrained BART-base-chinese encoder
#2. adding a linear layer to 1, substitute the embedding part of pretrained BART-base-english
#3. fine-tune params including linear, first layer attention
#4. fine-tune all params with decayed lr

n_epoch = 1
batch_size = 48
learning_rate = 0.01
max_length = 512
devices = jax.local_devices()
n_devices = jax.local_device_count()

def cross_entropy_loss(logits, labels):
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    exp_loss = np.take_along_axis(softmax_probs, labels[..., None], axis=-1)
    loss = -np.log(exp_loss)
    return np.sum(loss)

# 1. load params

ch_tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

params_ch = jax.tree_map(np.asarray, load_params('ch_trained_emb.dat'))
params_en = jax.tree_map(np.asarray, load_params('bart_params.dat'))

params = {
    'ch': {
        'embedding': params_ch['embedding'],
        'encoder_embed_layer_norm': params_ch['encoder_embed_layer_norm'],
        'encoder_embed_positions': params_ch['encoder_embed_positions'],
        'encoder_layers': params_ch['encoder_layers'],
    },
    'embedding': params_en['embedding'],
    'decoder_embed_positions': params_en['decoder_embed_positions'],
    'encoder_embed_layer_norm': params_en['encoder_embed_layer_norm'],
    'decoder_embed_layer_norm': params_en['decoder_embed_layer_norm'],
    'encoder_layers': params_en['encoder_layers'],
    'decoder_layers': params_en['decoder_layers'],
}

param_labels = {
    'ch': {
        'embedding': 'freeze',
        'encoder_embed_layer_norm': 'freeze',
        'encoder_embed_positions': 'freeze',
        'encoder_layers': 'train',
    },
    'encoder_embed_layer_norm': 'train',
    'encoder_layers': 'train',
    'embedding': 'freeze',
    'decoder_embed_positions': 'freeze',
    'decoder_embed_layer_norm': 'freeze',
    'decoder_layers': 'freeze',
}

replicated_params = jax.device_put_replicated(params, devices)

@jax.jit
@jax.value_and_grad
def stage1_loss_fn(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_nmt_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='num_devices')
def stage_1_batch_update(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = stage1_loss_fn(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')
    return grads, loss

@jax.jit
def stage1_eval_loss(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_nmt_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='num_devices')
def stage_1_batch_eval(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels):
    loss = stage1_eval_loss(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels)
    loss = jax.lax.pmean(loss, axis_name='num_devices')
    return loss




def device_split(arr):
    '''Splits the first axis of `arr` evenly across the number of devices.'''
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

def mask_1d_to_2d(mask_enc_1d, mask_dec_1d):
    mask_enc = device_split(np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None])
    mask_dec = device_split(np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None])
    mask_dec_enc = device_split(np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None])
    return mask_enc, mask_dec, mask_dec_enc

def evaluate(replicated_params):
    eval_input_ids, eval_mask_enc_1d, eval_decoder_input_ids, eval_mask_decoder_1d = process_one_dataset('dev/newsdev2017.zh', 'dev/newsdev2017.en')
    n_batches = len(eval_input_ids) // batch_size
    tqdm_eval_batch = trange(n_batches, desc='Batch', leave=False)
    epoch_loss = 0.
    for i in tqdm_eval_batch:
        src = device_split(input_ids[i * batch_size:(i + 1) * batch_size])
        dst = device_split(decoder_input_ids[i * batch_size:(i + 1) * batch_size])
        labels = device_split(onp.hstack(
            (decoder_input_ids[i * batch_size:(i + 1) * batch_size, 1:],
             np.ones((batch_size, 1), dtype=np.int32) * en_tokenizer.pad_token_id)))
        mask_enc, mask_dec, mask_dec_enc = mask_1d_to_2d(mask_enc_1d[i * batch_size:(i + 1) * batch_size],
                                                         mask_dec_1d[i * batch_size:(i + 1) * batch_size])
        loss = stage_1_batch_eval(replicated_params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels)
        batch_loss = jax.device_get(jax.tree_map(lambda x: x[0], loss)).item()
        epoch_loss += batch_loss
    epoch_loss /= n_batches
    return epoch_loss

def save_ckpt():
    params = jax.tree_map(lambda x: x[0], replicated_params)
    from flax.serialization import msgpack_serialize
    serialized_params = msgpack_serialize(params)
    with open('bart_stage1_keep_emb_ckpt.dat', 'wb') as f:
        f.write(serialized_params)


#stage 1
key = rand.PRNGKey(42)


# input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d, labels = load_dataset('dataset.npz')

input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d = process_one_dataset('wikimatrix21.zh', 'wikimatrix21.en')
# input_ids, mask_enc_1d = process_one_dataset('wikimatrix21.zh','zh')
# decoder_input_ids, mask_dec_1d = process_one_dataset('wikimatrix21.en', 'en')


# labels = onp.hstack((decoder_input_ids[:,1:], np.ones((len(input_ids), 1), dtype=np.int32) * en_tokenizer.pad_token_id))

n_sents = len(input_ids)

optimizer_scheme = {
    'train': optax.adam(learning_rate=learning_rate),
    'freeze': optax.set_to_zero(),
}

optimizer = optax.multi_transform(optimizer_scheme, param_labels)
optimizer = optax.chain(optax.adaptive_grad_clip(0.1, eps=0.001),
                                             optimizer)
opt_state = optimizer.init(params)

tqdm_epoch = trange(1, n_epoch + 1, desc='Epoch')
for _ in tqdm_epoch:
    epoch_loss = 0.
    eval_loss = math.inf

    n_batches = n_sents // batch_size
    key, subkey = rand.split(key)
    shuffled_indices = rand.permutation(subkey, n_sents)

    tqdm_batch = trange(n_batches, desc='Batch', leave=False)

    for i in tqdm_batch:
        batch = shuffled_indices[i*batch_size:(i+1)*batch_size]

        src = device_split(input_ids[batch])
        dst = device_split(decoder_input_ids[batch])
        labels = device_split(onp.hstack((decoder_input_ids[batch,1:], np.ones((len(batch), 1), dtype=np.int32) * en_tokenizer.pad_token_id)))

        mask_enc = device_split(np.einsum('bi,bj->bij', mask_enc_1d[batch], mask_enc_1d[batch])[:, None])
        mask_dec = device_split(np.tril(np.einsum('bi,bj->bij', mask_dec_1d[batch], mask_dec_1d[batch]))[:, None])
        mask_dec_enc = device_split(np.einsum('bi,bj->bij', mask_dec_1d[batch], mask_enc_1d[batch])[:, None])

        key, subkey = (lambda keys: (keys[0], keys[1:]))(rand.split(key, num=9))
        grads, loss = stage_1_batch_update(replicated_params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=subkey)

        grads = jax.tree_map(lambda x: x[0], grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        replicated_params = jax.device_put_replicated(params, devices)

        batch_loss = jax.tree_map(lambda x: x[0], loss).item()
        epoch_loss += batch_loss
        if i % 4 == 0:
            tqdm_batch.set_postfix({'batch loss': f'{batch_loss:.4f}'})

    epoch_loss /= n_batches
    tqdm_epoch.set_postfix({'epoch loss': f'{epoch_loss:.4f}'})

    new_eval_loss = evaluate(replicated_params)

    if new_eval_loss > eval_loss:
        break

    eval_loss = new_eval_loss

    save_ckpt()

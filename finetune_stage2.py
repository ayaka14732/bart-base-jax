import jax
import jax.nn as nn
import jax.numpy as np
import jax.random as rand
import math
import numpy as onp
from transformers import BertTokenizer, BartTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from tqdm import trange
import optax
import functools
from lib.fwd_nmt_transformer import fwd_nmt_transformer
from dataloader import process_one_dataset


#Procedure:
#1. load a pretrained BART-base-chinese encoder
#2. adding a linear layer to 1, substitute the embedding part of pretrained BART-base-english
#3. fine-tune params including linear, first layer attention
#4. fine-tune all params with decayed lr

n_epoch = 1
batch_size = 8
learning_rate = 0.01
max_length = 512
n_devices = jax.local_device_count()

def cross_entropy_loss(logits, labels):
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    exp_loss = np.take_along_axis(softmax_probs, labels[..., None], axis=-1)
    loss = -np.log(exp_loss)
    return np.sum(loss)

def load_params():
    from flax.serialization import msgpack_restore
    with open('bart_params.dat', 'rb') as f:
        b = f.read()
    params = msgpack_restore(b)
    params = jax.tree_map(np.asarray, params)  # NumPy array to JAX array
    return params

def load_ch_params():
    from flax.serialization import msgpack_restore
    with open('bart_ch_params.dat', 'rb') as f:
        b = f.read()
    ch_params = msgpack_restore(b)
    ch_params = jax.tree_map(np.asarray, ch_params)  # NumPy array to JAX array
    return ch_params



# 1. load params
ch_tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

from flax.serialization import msgpack_restore
with open('bart_stage1_fully_random_ckpt.dat', 'rb') as f:
    b = f.read()
params = msgpack_restore(b)
params = jax.tree_map(np.asarray, params)

replicated_params = jax.tree_map(lambda x: np.array([x] * n_devices), params)


def get_attn_values(params_dict):
    ret = []
    for k in params_dict:
        if k=='ff':
            continue
        if isinstance(params_dict[k],np.ndarray):
            ret.append(params_dict[k])
        else:
            ret.extend(get_attn_values(params_dict[k]))

@jax.jit
@jax.value_and_grad
def stage2_loss_fn(params,src,dst,mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_nmt_transformer(params,src,dst,mask_enc, mask_dec, mask_dec_enc)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    logits = nn.softmax(logits)
    loss = cross_entropy_loss(logits, labels) / len(labels)
    return loss

# https://github.com/google/jax/issues/9973#issuecomment-1073579382

@functools.partial(jax.pmap, axis_name='num_devices')
def stage_2_batch_update(params,src,dst,mask_enc, mask_dec, mask_dec_enc, labels):
    loss, grads = stage2_loss_fn(
        params,
        src,
        dst,
        mask_enc,
        mask_dec,
        mask_dec_enc,
        labels,
    )
    # .reshape(8, batch_size // 8, max_length)

    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    return grads, loss

@jax.jit
def stage2_eval_loss(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_nmt_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels) / len(labels)
    return loss


@functools.partial(jax.pmap, axis_name='num_devices')
def stage_2_batch_eval(params, src, dst, mask_enc, mask_dec, mask_dec_enc, labels):
    loss = stage2_eval_loss(
        params,
        src,
        dst,
        mask_enc,
        mask_dec,
        mask_dec_enc,
        labels,
    )
    # .reshape(8, batch_size // 8, max_length)

    loss = jax.lax.pmean(loss, axis_name='num_devices')
    return loss


def split(arr):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

def mask_1d_to_2d(mask_enc_1d, mask_dec_1d):
    mask_enc = split(np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None])
    mask_dec = split(np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None])
    mask_dec_enc = split(np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None])
    return mask_enc, mask_dec, mask_dec_enc

lm_head = params['embedding']['embedding'].T

#stage 1
key = rand.PRNGKey(42)

def eval(replicated_params):
    eval_input_ids, eval_mask_enc_1d, eval_decoder_input_ids, eval_mask_decoder_1d = process_one_dataset(
        'dev/newsdev2017.zh', 'dev/newsdev2017.en')
    n_batches = len(eval_input_ids) // batch_size
    tqdm_eval_batch = trange(n_batches, desc='Batch', leave=False)
    epoch_loss = 0.
    for i in tqdm_eval_batch:
        src = split(input_ids[i * batch_size:(i + 1) * batch_size])
        dst = split(decoder_input_ids[i * batch_size:(i + 1) * batch_size])
        labels = split(onp.hstack(
            (decoder_input_ids[i * batch_size:(i + 1) * batch_size, 1:],
             np.ones((batch_size, 1), dtype=np.int32) * en_tokenizer.pad_token_id)))

        mask_enc, mask_dec, mask_dec_enc = mask_1d_to_2d(mask_enc_1d[i * batch_size:(i + 1) * batch_size],
                                                         mask_dec_1d[i * batch_size:(i + 1) * batch_size])
        loss = stage_2_batch_eval(replicated_params, src, dst, mask_enc, mask_dec,
                                  mask_dec_enc, labels)
        batch_loss = jax.device_get(jax.tree_map(lambda x: x[0], loss)).item()
        epoch_loss += batch_loss
    epoch_loss /= n_batches
    return epoch_loss

# input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d, labels = load_dataset('dataset.npz')

# input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d = process_one_dataset('wikimatrix21.zh', 'wikimatrix21.en')
input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d = process_one_dataset('newscom21.zh', 'newscom21.en')


# input_ids, mask_enc_1d = process_one_dataset('wikimatrix21.zh','zh')
# decoder_input_ids, mask_dec_1d = process_one_dataset('wikimatrix21.en', 'en')


# labels = onp.hstack((decoder_input_ids[:,1:], np.ones((len(input_ids), 1), dtype=np.int32) * en_tokenizer.pad_token_id))

n_sents = len(input_ids)


# params = model.params
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

tqdm_epoch = trange(1, n_epoch + 1, desc='Epoch')
for _ in tqdm_epoch:
    epoch_loss = 0.
    eval_loss = math.inf

    n_batches = n_sents // batch_size
    key, subkey = rand.split(key)
    shuffled_indices = rand.permutation(subkey, n_sents)
    shuffled_indices = onp.asarray(shuffled_indices)

    tqdm_batch = trange(n_batches, desc='Batch', leave=False)

    for i in tqdm_batch:
        key, subkey = rand.split(key)
        batch = shuffled_indices[i * batch_size:(i + 1) * batch_size]

        src = split(input_ids[batch])
        dst = split(decoder_input_ids[batch])
        labels = split(onp.hstack(
            (decoder_input_ids[batch, 1:], np.ones((len(batch), 1), dtype=np.int32) * en_tokenizer.pad_token_id)))

        mask_enc, mask_dec, mask_dec_enc = mask_1d_to_2d(mask_enc_1d[batch], mask_dec_1d[batch])

        grads, loss = stage_2_batch_update(replicated_params, src, dst, mask_enc, mask_dec,
                                           mask_dec_enc, labels)

        grads = jax.device_get(jax.tree_map(lambda x: x[0], grads))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        replicated_params = jax.tree_map(lambda x: np.array([x] * n_devices), params)

        batch_loss = jax.device_get(jax.tree_map(lambda x: x[0], loss)).item()
        epoch_loss += batch_loss
        if i % 4 == 0:
            tqdm_batch.set_postfix({'batch loss': f'{batch_loss:.4f}'})

    epoch_loss /= n_batches
    tqdm_epoch.set_postfix({'epoch loss': f'{epoch_loss:.4f}'})

    new_eval_loss = eval(replicated_params, replicated_other_params)

    if new_eval_loss > eval_loss:
        break

    eval_loss = new_eval_loss

#save stage 1 checkpoint
params = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))

from flax.serialization import msgpack_serialize
serialized_params = msgpack_serialize(params)
with open('bart_stage2_random_enc_ckpt.dat', 'wb') as f:
    f.write(serialized_params)



import jax
import jax.nn as nn
import jax.numpy as np
import jax.random as rand
import math
import numpy as onp
from transformers import BertTokenizer, BartTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from tqdm import trange
import optax
from lib.fwd_nmt_transformer import fwd_nmt_transformer

#Procedure:
#1. load a pretrained BART-base-chinese encoder
#2. adding a linear layer to 1, substitute the embedding part of pretrained BART-base-english
#3. fine-tune params including linear, first layer attention
#4. fine-tune all params with decayed lr

n_epoch = 10
batch_size = 112
learning_rate = 0.005
max_length = 512


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
# model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
# ch_parameters = recursive_turn_to_jnp(dict(model.model.named_parameters()))
ch_params = load_ch_params()

w_initializer = jax.nn.initializers.orthogonal()
b_initializer = jax.nn.initializers.uniform(1/math.sqrt(768))
linear_params = {'kernel':w_initializer(rand.PRNGKey(42), (768, 768), np.float32),'bias':b_initializer(rand.PRNGKey(42), (768), np.float32)}

en_params = load_params()


params = {'added_linear':linear_params, 'first_attn':en_params['encoder_layers'][0]['self_attn']}
other_params = {**en_params,'ch':ch_params}

def load_dataset(filename):
    z = onp.load(filename)

    input_ids = z['input_ids']
    attention_mask = z['attention_mask']
    decoder_input_ids = z['decoder_input_ids']
    decoder_attention_mask = z['decoder_attention_mask']
    n_sents = len(input_ids)
    labels = onp.hstack((decoder_input_ids[:,1:], np.ones((n_sents, 1), dtype=np.int32) * ch_tokenizer.pad_token_id))

    return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels

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
def stage1_loss_fn(params,other_params,src,dst,mask_enc, mask_dec, mask_dec_enc, labels):
    other_params['encoder_layers'][0]['self_attn'] = params['first_attn']
    fwd_params = {'added_linear':params['added_linear'],**other_params}
    outputs = fwd_nmt_transformer(fwd_params,src,dst,mask_enc, mask_dec, mask_dec_enc)
    loss = cross_entropy_loss(outputs.logits, labels) / len(labels)
    return loss

@jax.jit
@jax.value_and_grad
def stage2_loss_fn(params,src,dst,mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_nmt_transformer(params,src,dst,mask_enc, mask_dec, mask_dec_enc)
    loss = cross_entropy_loss(outputs.logits, labels) / len(labels)
    return loss

# https://github.com/google/jax/issues/9973#issuecomment-1073579382

lm_head = en_params['embedding']['embedding'].T

#stage 1
key = rand.PRNGKey(42)
input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = load_dataset('dataset.npz')
assert input_ids.shape[1] == attention_mask.shape[1] == decoder_input_ids.shape[1] == decoder_attention_mask.shape[1] == labels.shape[1] == max_length
n_sents = len(input_ids)

# params = model.params
optimizer = optax.lamb(learning_rate=learning_rate)
opt_state = optimizer.init(params)


tqdm_epoch = trange(1, n_epoch + 1, desc='Epoch')
for _ in tqdm_epoch:
    epoch_loss = 0.

    n_batches = n_sents // batch_size
    key, subkey = rand.split(key)
    shuffled_indices = rand.permutation(subkey, n_sents)

    tqdm_batch = trange(n_batches, desc='Batch', leave=False)

    for i in tqdm_batch:
        key, subkey = rand.split(key)
        batch = shuffled_indices[i*batch_size:(i+1)*batch_size]

        loss, grads = stage1_loss_fn(
            params,
            src,
            dst,
            mask_enc,
            mask_dec,
            mask_dec_enc,
            labels[batch,],
        )
        # .reshape(8, batch_size // 8, max_length)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        batch_loss = loss.item()
        epoch_loss += batch_loss

    epoch_loss /= n_batches
    tqdm_epoch.set_postfix({'epoch loss': f'{epoch_loss:.4f}'})


#stage 2
input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = load_dataset('dataset.npz')
assert input_ids.shape[1] == attention_mask.shape[1] == decoder_input_ids.shape[1] == decoder_attention_mask.shape[1] == labels.shape[1] == max_length
n_sents = len(input_ids)

params = {'added_linear':params['added_linear'],**other_params}

optimizer = optax.lamb(learning_rate=0.001)
opt_state = optimizer.init(params)


tqdm_epoch = trange(1, n_epoch + 1, desc='Epoch')
for _ in tqdm_epoch:
    epoch_loss = 0.

    n_batches = n_sents // batch_size
    key, subkey = rand.split(key)
    shuffled_indices = rand.permutation(subkey, n_sents)

    tqdm_batch = trange(n_batches, desc='Batch', leave=False)

    for i in tqdm_batch:
        key, subkey = rand.split(key)
        batch = shuffled_indices[i*batch_size:(i+1)*batch_size]

        loss, grads = stage1_loss_fn(
            params,
            src,
            dst,
            mask_enc,
            mask_dec,
            mask_dec_enc,
            labels[batch,],
        )
        # .reshape(8, batch_size // 8, max_length)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        batch_loss = loss.item()
        epoch_loss += batch_loss

    epoch_loss /= n_batches
    tqdm_epoch.set_postfix({'epoch loss': f'{epoch_loss:.4f}'})


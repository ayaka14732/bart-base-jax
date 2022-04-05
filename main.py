import jax
import jax.nn as nn
import jax.numpy as np
import numpy as onp
from transformers import BartTokenizer

from lib.fwd_transformer import fwd_transformer

# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

def load_params():
    from flax.serialization import msgpack_restore
    with open('bart_params.dat', 'rb') as f:
        b = f.read()
    # with open('bart_stage1_fully_random_ckpt.dat', 'rb') as f:
    #     b = f.read()
    params = msgpack_restore(b)
    params = jax.tree_map(np.asarray, params)  # NumPy array to JAX array
    return params


params = load_params()

lm_head = params['embedding']['embedding'].T

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# sentences = ['Can you see the beautiful flowers <mask> alongside the track?']

# sentences = ['Can you see the beautiful flowers blooming alongside the track?']
#
# batch = tokenizer(sentences, return_tensors='jax',padding=True)

# src = batch.input_ids
# mask_enc_1d = batch.attention_mask.astype(np.bool_)

from dataloader import process_one_dataset
input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d = process_one_dataset('newscom21.zh', 'newscom21.en',1)
src = decoder_input_ids[1:]
dst = decoder_input_ids

i = 1
# dst = np.zeros((len(sentences), 1), dtype=np.int32)
# dst = np.asarray(np.asarray([0])+src[:-1],dtype=np.int32)
# print(tokenizer.batch_decode(dst, skip_special_tokens=False))

mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
mask_dec = np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
mask_dec_enc = np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

labels = onp.hstack(
            (decoder_input_ids,
             np.ones((len(input_ids), 1), dtype=np.int32) * tokenizer.pad_token_id))

y = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
a = y @ lm_head

exp_logits = np.exp(a)
softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
exp_loss = np.take_along_axis(softmax_probs, labels[..., None], axis=-1)
loss = -np.log(exp_loss)
print(loss)

a = np.argmax(softmax_probs[:, -1], axis=-1)

# while True:
#     # mask_dec_1d = np.ones((len(sentences), i), dtype=np.bool_)
#
#     mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
#     mask_dec = np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
#     mask_dec_enc = np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]
#
#     y = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
#
#     a = nn.softmax(y @ lm_head)
#     a = np.argmax(a[:, -1], axis=-1)
#
#     i += 1
#     dst = np.hstack((dst, a[..., None]))
#
#     if np.all(a == 2):
#         break
# print(tokenizer.batch_decode(src, skip_special_tokens=False))
# print(tokenizer.batch_decode(dst, skip_special_tokens=False))

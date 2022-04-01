import jax
import jax.nn as nn
import jax.numpy as np
import jax.random as rand
import torch
import math
from transformers import BartTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

from lib.fwd_transformer import fwd_transformer



#Procedure:
#1. load a pretrained BART-base-chinese encoder
#2. adding a linear layer to 1, substitute the embedding part of pretrained BART-base-english
#3. fine-tune params including linear, first layer attention
#4. fine-tune all params with decayed lr

def recursive_turn_to_jnp(params):  #turn dict of (list of) pytorch tensor into JAX array
    for k in params:
        if isinstance(params[k],torch.nn.parameter.Parameter):
            params[k] = np.asarray(params[k].detach().numpy())
        else:
            recursive_turn_to_jnp(params[k])


# 1.
ch_tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
ch_parameters = recursive_turn_to_jnp(dict(model.model.named_parameters()))
model = BartForConditionalGeneration.from_pretrained('')

w_initializer = jax.nn.initializers.orthogonal()
b_initializer = jax.nn.initializers.uniform(1/math.sqrt(768))
linear_params = {'kernel':w_initializer(rand.PRNGKey(42), (768, 768), np.float32),'bias':b_initializer(rand.PRNGKey(42), (768), jnp.float32)}








# https://github.com/google/jax/issues/9973#issuecomment-1073579382

lm_head = params['embedding']['embedding'].T

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

sentences = ['Can you see the beautiful flowers <mask> alongside the track?']
batch = tokenizer(sentences, return_tensors='jax')

src = batch.input_ids
mask_enc_1d = batch.attention_mask.astype(np.bool_)

i = 1
dst = np.zeros((len(sentences), 1), dtype=np.int32)

while True:
    mask_dec_1d = np.ones((len(sentences), i), dtype=np.bool_)

    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    y = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)

    a = nn.softmax(y @ lm_head)
    a = np.argmax(a[:, -1], axis=-1)

    i += 1
    dst = np.hstack((dst, a[..., None]))
    dst

    if np.all(a == 2):
        break

print(tokenizer.batch_decode(dst, skip_special_tokens=True))

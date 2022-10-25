import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
from transformers import BertTokenizer, FlaxBartModel

from lib.param_utils.random_init_embed import random_init_embed
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key

tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
model = FlaxBartModel.from_pretrained('fnlp/bart-base-chinese', from_pt=True)

key = seed2key(42)

##########

token_new_to_token_id = {}

with open('vocab_mapping.txt', encoding='utf-8') as f:
    for line in f:
        token_new, token_id = line.rstrip('\n').rsplit(' ', 1)
        token_id = int(token_id)
        token_new_to_token_id[token_new] = token_id

##########

with open('add_token.txt', encoding='utf-8') as f:
    new_chars = f.read().rstrip('\n')

##########

num_of_unused = sum(1 for token in token_new_to_token_id if token.startswith('[unused'))
assert num_of_unused == 99

new_chars_a = new_chars[:num_of_unused]
new_chars_b = new_chars[num_of_unused:]

##########

emb_old = list(model.params['shared']['embedding'])
emb_new = list(random_init_embed(key, len(new_chars_b)))

##########

token_new_to_emb = {}

unused_idx = 1
new_chars_iter = iter(new_chars_a)

for token_new, token_id in token_new_to_token_id.items():
    if token_new.startswith('[unused'):
        token_id = token_new_to_token_id[f'[unused{unused_idx}]']
        new_char = next(new_chars_iter)
        token_new_to_emb[new_char] = emb_old[token_id]
        unused_idx += 1
    else:
        token_new_to_emb[token_new] = emb_old[token_id]

for i, token_new in enumerate(new_chars_b):
    token_new_to_emb[token_new] = emb_new[i]

##########

vocab = []
emb_all = []

for i, (token_new, emb) in enumerate(token_new_to_emb.items()):
    assert isinstance(token_new, str)
    assert emb.shape[0] == 768
    assert token_new != '\n'
    vocab.append(token_new)
    emb_all.append(emb)

emb_all = np.vstack(emb_all)

##########

with open('vocab-bart-base-cantonese.txt') as f:
    for token in vocab:
        print(token, file=f)

save_params(emb_all, 'embed_params.dat')

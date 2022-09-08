import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
import math
import random

from lib.param_utils.save_params import save_params
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser

tokeniser = CharBasedTokeniser.from_vocab_file('vocab.txt')

list_tokenised_mandarin = []
list_tokenised_hokkien = []

with open('lib/twblg/data.tsv', encoding='utf-8') as f:
    lines = list(f)

random.seed(42)
random.shuffle(lines)

for line in lines:
    mandarin, hokkien = line.rstrip('\n').split('\t')
    tokenised_mandarin = tokeniser.tokenise_sentence(mandarin)
    tokenised_hokkien = tokeniser.tokenise_sentence(hokkien)
    list_tokenised_mandarin.append(tokenised_mandarin)
    list_tokenised_hokkien.append(tokenised_hokkien)

len_mandarin = max(len(l) for l in list_tokenised_mandarin)
len_hokkien = max(len(l) for l in list_tokenised_hokkien)

pad_token = tokeniser.pad_token

list_tokenised_padded_mandarin = []
mask_mandarin = []

for l in list_tokenised_mandarin:
    content_len = len(l)
    pad_len = len_mandarin - content_len
    list_tokenised_padded_mandarin.append([*l, *((pad_token,) * pad_len)])
    mask_mandarin.append([*((1,) * content_len), *((0,) * pad_len)])

list_tokenised_padded_hokkien = []
mask_hokkien = []

for l in list_tokenised_hokkien:
    content_len = len(l)
    pad_len = len_hokkien - content_len
    list_tokenised_padded_hokkien.append([*l, *((pad_token,) * pad_len)])
    mask_hokkien.append([*((1,) * content_len), *((0,) * pad_len)])

src = np.array(list_tokenised_padded_mandarin, dtype=np.uint16)
mask_enc_1d = np.array(mask_mandarin, dtype=np.bool_)
dst = np.array(list_tokenised_padded_hokkien, dtype=np.uint16)
mask_dec_1d = np.array(mask_hokkien, dtype=np.bool_)

len_dataset = len(src)
len_train = math.floor(len_dataset * 0.8)
len_dev = math.floor(len_dataset * 0.1)
len_test = len_dataset - len_train - len_dev

# train

save_params({
    'src': src[:len_train],
    'mask_enc_1d': mask_enc_1d[:len_train],
    'dst': dst[:len_train],
    'mask_dec_1d': mask_dec_1d[:len_train],
}, 'train.dat')

# dev

save_params({
    'src': src[len_train:len_train+len_dev],
    'mask_enc_1d': mask_enc_1d[len_train:len_train+len_dev],
    'dst': dst[len_train:len_train+len_dev],
    'mask_dec_1d': mask_dec_1d[len_train:len_train+len_dev],
}, 'dev.dat')

# test

save_params({
    'src': src[len_train+len_dev:],
    'mask_enc_1d': mask_enc_1d[len_train+len_dev:],
    'dst': dst[len_train+len_dev:],
    'mask_dec_1d': mask_dec_1d[len_train+len_dev:],
}, 'test.dat')

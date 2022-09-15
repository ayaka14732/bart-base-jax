import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
import math
import random

from lib.param_utils.save_params import save_params
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser

tokeniser = CharBasedTokeniser(vocab='vocab.txt')

with open('lib/twblg/data.tsv', encoding='utf-8') as f:
    lines = list(f)

random.seed(42)
random.shuffle(lines)

mandarins = []
hokkiens = []

for line in lines:
    _, _, mandarin, hokkien = line.rstrip('\n').split('\t')
    mandarins.append(mandarin)
    hokkiens.append(hokkien)

max_len_mandarin = max(len(l) for l in mandarins) + 2  # 2: [BOS], [EOS]
max_len_hokkien = max(len(l) for l in hokkiens) + 2

tokenised_mandarin = tokeniser(mandarins, return_tensors='np', max_length=max_len_mandarin, padding='max_length', truncation=True)
tokenised_hokkien = tokeniser(hokkien, return_tensors='np', max_length=max_len_hokkien, padding='max_length', truncation=True)

src = tokenised_mandarin.input_ids.astype(np.uint16)
mask_enc_1d = tokenised_mandarin.attention_mask.astype(np.bool_)
dst = tokenised_hokkien.input_ids.astype(np.uint16)
mask_dec_1d = tokenised_hokkien.attention_mask.astype(np.bool_)

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

import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np

from lib.param_utils.save_params import save_params

from lib.twblg.CharBasedTokeniser import CharBasedTokeniser

tokeniser = CharBasedTokeniser.from_vocab_file('vocab.txt')

list_tokenised_mandarin = []
list_tokenised_hokkien = []

with open('twblg/data.tsv', encoding='utf-8') as f:
    for line in f:
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

save_params({
    'src': src,
    'mask_enc_1d': mask_enc_1d,
    'dst': dst,
    'mask_dec_1d': mask_dec_1d,
}, 'dataset.dat')

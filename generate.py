import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
from transformers import BartConfig

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.simple_dataloader.SimpleDataLoader import SimpleDataLoader
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser
from lib.twblg.fwd_transformer_encoder_part import fwd_transformer_encoder_part

config = BartConfig.from_pretrained(
    'fnlp/bart-base-chinese',
    bos_token_id=2,
    eos_token_id=3,
    vocab_size=7697,
)

params = load_params('helpful-bird-17.dat')
params = jax.tree_map(np.asarray, params)
generator = Generator(params, config=config)

tokeniser = CharBasedTokeniser.from_vocab_file('vocab.txt')

data_loader = SimpleDataLoader('dataset.dat', batch_size=1, shuffle=False)


i = 0
for batch in data_loader:
    i+=1
    if i < 200:
        continue
    if i >= 225:
        break

    src = batch.src
    mask_enc = batch.mask_enc
    mask_enc_1d = batch.mask_enc_1d

    encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
    generated_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=100, bos_token_id=2, pad_token_id=0, eos_token_id=3, decoder_start_token_id=2)

    print('Src:', tokeniser.detokenise_sentence(src[0].tolist()))
    print('Gld:', tokeniser.detokenise_sentence(batch.dst[0].tolist()))
    print('Out:', tokeniser.detokenise_sentence(generated_ids[0].tolist()))
    print()

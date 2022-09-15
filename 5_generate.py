import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
from StarCC import PresetConversion
from transformers import BartConfig

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.simple_dataloader.SimpleDataLoader import SimpleDataLoader
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser
from lib.twblg.fwd_transformer_encoder_part import fwd_transformer_encoder_part

trad = lambda x: PresetConversion(src='cn', dst='tw', with_phrase=False)(x).replace('代志', '代誌')

config = BartConfig.from_pretrained(
    'fnlp/bart-base-chinese',
    bos_token_id=2,
    eos_token_id=3,
    vocab_size=6995,
)

params = load_params('serene-disco-36.dat')
params = jax.tree_map(np.asarray, params)
generator = Generator(params, config=config)

tokeniser = CharBasedTokeniser.from_vocab_file('vocab.txt')

data_loader = SimpleDataLoader('test.dat', batch_size=1, shuffle=False)

for batch in data_loader:
    src = batch.src
    mask_enc = batch.mask_enc
    mask_enc_1d = batch.mask_enc_1d

    encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
    generated_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=7, max_length=100, bos_token_id=2, pad_token_id=0, eos_token_id=3, decoder_start_token_id=2, num_return_sequences=3)

    print('Src:', trad(tokeniser.detokenise_sentence(src[0].tolist())))
    print('Gld:', trad(tokeniser.detokenise_sentence(batch.dst[0].tolist())).replace('/', ''))
    print('Ot1:', trad(tokeniser.detokenise_sentence(generated_ids[0].tolist())).replace('/', ''))
    print('Ot2:', trad(tokeniser.detokenise_sentence(generated_ids[1].tolist())).replace('/', ''))
    print('Ot3:', trad(tokeniser.detokenise_sentence(generated_ids[2].tolist())).replace('/', ''))
    print()

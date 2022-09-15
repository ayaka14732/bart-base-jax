import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax  # ; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
from StarCC import PresetConversion
from transformers import BartConfig

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.simple_dataloader.SimpleDataLoader import SimpleDataLoader
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser
from lib.twblg.fwd_transformer_encoder_part import fwd_transformer_encoder_part

batch_size = 512
num_return_sequences = 3

# trad = lambda x: PresetConversion(src='cn', dst='tw', with_phrase=False)(x).replace('代志', '代誌')
trad = lambda x: x

config = BartConfig.from_pretrained(
    'fnlp/bart-base-chinese',
    bos_token_id=2,
    eos_token_id=3,
    vocab_size=6995,
)

params = load_params('peachy-pine-38.dat')
params = jax.tree_map(np.asarray, params)
generator = Generator(params, config=config)

fwd_transformer_encoder_part = jax.jit(fwd_transformer_encoder_part)

def main():
    tokeniser = CharBasedTokeniser(vocab='vocab.txt')

    data_loader = SimpleDataLoader('test.dat', batch_size=batch_size, shuffle=False)

    for batch in data_loader:
        src = batch.src
        mask_enc = batch.mask_enc
        mask_enc_1d = batch.mask_enc_1d

        encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
        generated_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=7, max_length=100, bos_token_id=2, pad_token_id=0, eos_token_id=3, decoder_start_token_id=2, num_return_sequences=num_return_sequences)

        for src_, dst_, generated_ids_ in zip(src, batch.dst, generated_ids.reshape(batch_size, num_return_sequences, -1)):
            print('輸入：', trad(tokeniser.decode(src_)), sep='')
            print('預期：', trad(tokeniser.decode(dst_)).replace('/', ''), sep='')
            for generated_id_ in tokeniser.batch_decode(generated_ids_):
                print('輸出：', trad(generated_id_).replace('/', ''), sep='')
            print()

if __name__ == '__main__':
    main()

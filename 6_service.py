import jax; jax.config.update('jax_platforms', 'cpu')

import cherrypy
import jax.numpy as np
import logging
from StarCC import PresetConversion
from transformers import BartConfig

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser
from lib.twblg.fwd_transformer_encoder_part import fwd_transformer_encoder_part

# disable logging
logging.getLogger('cherrypy').propagate = False

batch_size = 512
num_return_sequences = 3

simp = PresetConversion(src='tw', dst='cn', with_phrase=False)
trad_ = PresetConversion(src='cn', dst='tw', with_phrase=False)
trad = lambda s: trad_(s).replace('代志', '代誌')

config = BartConfig.from_pretrained(
    'fnlp/bart-base-chinese',
    bos_token_id=2,
    eos_token_id=3,
    vocab_size=7153,
)

params = load_params('peachy-pine-38.dat')
params = jax.tree_map(np.asarray, params)
generator = Generator(params, config=config)

fwd_transformer_encoder_part = jax.jit(fwd_transformer_encoder_part)

tokeniser = CharBasedTokeniser(vocab='vocab.txt')

def translate(s):
    s = simp(s)
    inputs = tokeniser([s], return_tensors='jax')

    src = inputs.input_ids.astype(np.uint16)
    mask_enc_1d = inputs.attention_mask.astype(np.bool_)
    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

    encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
    generated_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=7, max_length=768, bos_token_id=2, pad_token_id=0, eos_token_id=3, decoder_start_token_id=2, num_return_sequences=num_return_sequences)

    generated_sentences = tokeniser.batch_decode(generated_ids)
    generated_sentences = list({trad(s.replace('/', '')): None for s in generated_sentences})
    return generated_sentences

class Server:
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self, s):
        assert isinstance(s, str) and s
        return translate(s)

if __name__ == '__main__':
    cherrypy.config.update({
        'environment': 'production',
        'log.screen': False,
        'server.socket_host': '127.0.0.1',
        'server.socket_port': 31345,
        'show_tracebacks': False,
        'server.thread_pool': 1,
    })
    cherrypy.quickstart(Server())

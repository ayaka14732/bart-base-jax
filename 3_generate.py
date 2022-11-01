import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as np
from transformers import BartConfig, BartTokenizer, BertTokenizer
from lib.Generator import Generator

from lib.param_utils.load_params import load_params
from lib.en_kfw_nmt.fwd_transformer_encoder_part import fwd_transformer_encoder_part

params = load_params('params_merged.dat')
params = jax.tree_map(np.asarray, params)

tokenizer_en = BartTokenizer.from_pretrained('facebook/bart-base')
tokenizer_yue = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

config = BartConfig.from_pretrained('Ayaka/bart-base-cantonese')
generator = Generator({'embedding': params['decoder_embedding'], **params}, config=config)

# generate

sentences = [
    'Fire!',
    'Are you feeling unwell?',
    'How long have you waited?',
    'muscular dystrophy',
    'You guys have no discipline, how can you be part of the disciplined services?',
    'Set A for takeaway.',
]
inputs = tokenizer_en(sentences, return_tensors='jax', max_length=20, padding='max_length', truncation=True)
src = inputs.input_ids.astype(np.uint16)
mask_enc_1d = inputs.attention_mask.astype(np.bool_)
mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
generate_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5)

decoded_sentences = tokenizer_yue.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(decoded_sentences)

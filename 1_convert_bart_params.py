import jax; jax.config.update('jax_platforms', 'cpu')
from transformers import FlaxBartModel

from lib.param_utils.flax2jax import flax2jax
from lib.param_utils.save_params import save_params

model_en = FlaxBartModel.from_pretrained('facebook/bart-base')
params_en = flax2jax(model_en.params)

model_yue = FlaxBartModel.from_pretrained('Ayaka/bart-base-cantonese')
params_yue = flax2jax(model_yue.params)

params = {
    'encoder_embedding': params_en['embedding'],
    'encoder_embed_positions': params_en['encoder_embed_positions'],
    'encoder_embed_layer_norm': params_en['encoder_embed_layer_norm'],
    'encoder_layers': params_en['encoder_layers'],
    'decoder_embedding': params_yue['embedding'],
    'decoder_embed_positions': params_yue['decoder_embed_positions'],
    'decoder_embed_layer_norm': params_yue['decoder_embed_layer_norm'],
    'decoder_layers': params_yue['decoder_layers'],
    'lm_head': params_yue['embedding']['embedding'].T,
}

save_params(params, 'params_merged.dat')

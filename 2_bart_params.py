import jax; jax.config.update('jax_platforms', 'cpu')

from transformers import FlaxBartModel

from lib.param_utils.flax2jax import flax2jax
from lib.param_utils.load_params import load_params
from lib.param_utils.save_params import save_params

model = FlaxBartModel.from_pretrained('fnlp/bart-base-chinese', from_pt=True)
params = flax2jax(model.params)
params['embedding']['embedding'] = load_params('embed_params.dat')
params['lm_head'] = params['embedding']['embedding'].T
save_params(params, 'untrained_params.dat')

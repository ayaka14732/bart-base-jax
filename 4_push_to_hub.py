import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as np
import tempfile
from transformers import BartConfig, BartModel, FlaxBartModel

from lib.param_utils.load_params import load_params
from lib.param_utils.jax2flax import jax2flax

params = load_params('electric-glade-5-7-61440.dat')
vocab_size, *_ = params['embedding']['embedding'].shape

params_flax = jax2flax(params)
params_flax = jax.tree_map(np.asarray, params_flax)

config = BartConfig.from_pretrained('fnlp/bart-base-chinese', vocab_size=12660)
model_flax = FlaxBartModel(config=config)

model_flax.params = params_flax
model_flax.config.vocab_size
model_flax.push_to_hub('Ayaka/bart-base-cantonese')

with tempfile.TemporaryDirectory() as tmpdirname:
    model_flax.save_pretrained(tmpdirname)
    model_pt = BartModel.from_pretrained(tmpdirname, from_flax=True)

model_pt.push_to_hub('Ayaka/bart-base-cantonese')

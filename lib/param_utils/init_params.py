import jax.random as rand
import numpy as np
from transformers import BartConfig, FlaxBartModel

from .flax2jax import flax2jax

_config = BartConfig.from_pretrained('facebook/bart-base')

_minval = np.iinfo(np.int32).min
_maxval = np.iinfo(np.int32).max

def init_params(key: rand.KeyArray) -> dict:
    key = rand.PRNGKey(42)
    seed = rand.randint(key, (), _minval, _maxval).item()

    model_flax = FlaxBartModel(config=_config, seed=seed)
    model_jax = flax2jax(model_flax.params)

    return model_jax

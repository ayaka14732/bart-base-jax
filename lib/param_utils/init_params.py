import jax.random as rand
from transformers import BartConfig, FlaxBartModel

from .flax2jax import flax2jax
from ..random.wrapper import key2seed

_config = BartConfig.from_pretrained('facebook/bart-base')

def init_params(key: rand.KeyArray) -> dict:
    seed = key2seed(key)
    model_flax = FlaxBartModel(config=_config, seed=seed)
    model_jax = flax2jax(model_flax.params)
    return model_jax

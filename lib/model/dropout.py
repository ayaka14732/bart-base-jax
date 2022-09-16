import jax.random as rand
from jaxtyping import Array, jaxtyped
from typeguard import typechecked as typechecker

from ..random.wrapper import KeyArray

# @jaxtyped
# @typechecker
def dropout(key: KeyArray, x: Array['*dims'], keep_rate: float=0.9) -> Array['*dims']:
    assert 0. <= keep_rate <= 1.
    y = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    return y

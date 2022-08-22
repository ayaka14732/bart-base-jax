from jax._src.random import KeyArray
import jax.random as rand
from jaxtyping import f as F, jaxtyped
from typeguard import typechecked as typechecker

@jaxtyped
@typechecker
def dropout(key: KeyArray, x: F['*dims'], keep_rate: float=0.9) -> F['*dims']:
    assert 0. <= keep_rate <= 1.
    y = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    return y

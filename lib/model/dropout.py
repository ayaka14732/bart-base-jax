from jax._src.random import KeyArray
from jaxtyping import f as F, jaxtyped
from typeguard import typechecked as typechecker

from ..debug.log import log_shape
from ..random.wrapper import bernoulli

@jaxtyped
@typechecker
def dropout(key: KeyArray, x: F['*dims']) -> F['*dims']:
    keep_rate = 0.9

    log_shape('x', x)

    y = x * bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate

    log_shape('y', y)

    return y

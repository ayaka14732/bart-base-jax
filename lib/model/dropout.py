import jax.numpy as np
import jax.random as rand

from ..debug.log import log_shape

def dropout(key: rand.KeyArray, x: np.ndarray):
    keep_rate = 0.9

    log_shape('x', x)

    y = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate

    log_shape('y', y)

    return y

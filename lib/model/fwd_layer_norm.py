import jax.numpy as np

from ..debug.log import log_shape

def fwd_layer_norm(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    scale: np.ndarray = params['scale']  # array
    bias: np.ndarray = params['bias']  # array

    eps = 1e-5

    log_shape('scale', scale)
    log_shape('bias', bias)

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)

    log_shape('mean', mean)
    log_shape('var', var)

    y = ((x - mean) / np.sqrt(var + eps)) * scale + bias

    log_shape('y', y)

    return y

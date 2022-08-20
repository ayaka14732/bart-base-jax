import jax.numpy as np

from ..debug.log import log_shape

def fwd_linear(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    kernel: np.ndarray = params['kernel']  # array
    bias: np.ndarray = params['bias']  # array

    log_shape('kernel', kernel)
    log_shape('bias', bias)

    y = np.dot(x, kernel) + bias

    log_shape('y', y)

    return y

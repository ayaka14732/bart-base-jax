import jax.numpy as np

from ..debug.log import log_shape

def fwd_embedding(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    embedding: np.ndarray = params['embedding']  # array

    log_shape('embedding', embedding)

    y = embedding[x]

    log_shape('y', y)

    return y

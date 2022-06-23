import jax.numpy as np

def fwd_linear(params: dict, x: np.ndarray) -> np.ndarray:
    if not 'bias' in params:
        # params
        kernel: np.ndarray = params['kernel']  # array

        return np.dot(x, kernel)

    else:
        # params
        kernel: np.ndarray = params['kernel']  # array
        bias: np.ndarray = params['bias']  # array

        return np.dot(x, kernel) + bias

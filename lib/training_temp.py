import jax.numpy as np

def device_split(arr: np.ndarray, n_devices: int) -> np.ndarray:
    '''Splits the first axis of `arr` evenly across the number of devices.'''
    batch_size, *shapes = arr.shape
    return arr.reshape(n_devices, batch_size // n_devices, *shapes)

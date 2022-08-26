from jaxtyping import Array, jaxtyped
from typeguard import typechecked as typechecker

@jaxtyped
@typechecker
def device_split(arr: Array['...'], n_devices: int) -> Array['...']:
    '''Splits the first axis of `arr` evenly across the number of devices.'''
    batch_size, *shapes = arr.shape
    return arr.reshape(n_devices, batch_size // n_devices, *shapes)

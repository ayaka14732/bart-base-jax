import jax
from jaxtyping import Array, Shaped as S, jaxtyped
import numpy as onp
from typeguard import typechecked as typechecker

@jaxtyped
@typechecker
def device_split(a: S[onp.ndarray, '...']) -> S[Array, '...']:
    local_devices = jax.local_devices()
    n_local_devices = jax.device_count()

    '''Splits the first axis of `a` evenly across the number of devices.'''
    batch_size, *shapes = a.shape
    a = a.reshape(local_devices, batch_size // n_local_devices, *shapes)
    b = jax.device_put_sharded(tuple(a), devices=local_devices)
    return b

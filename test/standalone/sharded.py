import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'

import jax
import numpy as onp

jax.config.update('jax_platforms', 'cpu')

devices = jax.devices()
n_devices = jax.device_count()
assert n_devices == 8

a = onp.zeros((8, 36, 128))
b = jax.device_put_sharded(tuple(a), devices=devices)

print('Original shape:', a.shape)
print('Sharded shape:', b.shape)
print('Sharded devices:', [buffer.device() for buffer in b.device_buffers])

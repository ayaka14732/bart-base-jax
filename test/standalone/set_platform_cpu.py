import jax
import jax.numpy as np

jax.config.update('jax_platforms', 'cpu')

a = np.ones((4, 8))
assert 'CPU' in str(a.device())

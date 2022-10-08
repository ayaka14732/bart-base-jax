import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np
from jaxtyping import Array, Float as F, jaxtyped
from typeguard import check_type, typechecked as typechecker

@jax.jit
@jaxtyped
@typechecker
@jax.jit  # simulate nested `jax.jit`
def f(a: F[Array, 'a b c d e'], b: F[Array, 'b c e f g']) -> F[Array, 'a g d e f']:
    c = a + 1.
    check_type('c', c, F[Array, 'a b c d e'])
    d = np.einsum('abcde,bcefg->agdef', c, b)
    check_type('d', d, F[Array, 'a g d e f'])
    e = np.cos(d)
    check_type('e', e, F[Array, 'a g d e f'])
    return e

if __name__ == '__main__':
    a = np.ones((2, 3, 4, 5, 6))
    b = np.ones((3, 4, 6, 7, 8))
    f(a, b)
    f(a, b)
    f(a, b)

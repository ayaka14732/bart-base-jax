import jax
import jax.numpy as np
from jaxtyping import f as F, jaxtyped
from typeguard import check_type, typechecked as typechecker

jax.config.update('jax_platforms', 'cpu')

@jaxtyped
@typechecker
def f1(a: F['a b c d e'], b: F['b c e f g']) -> F['a g d e f']:
    c = a + 1.
    check_type('c', c, F['a b c d e'])
    d = np.einsum('abcde,bcefg->agdef', c, b)
    check_type('d', d, F['a g d e f'])
    e = np.cos(d)
    check_type('e', e, F['a g d e f'])
    return e

@jaxtyped
@typechecker
def f2(a: F['a b c d e'], b: F['b c e f g']) -> F['a g d e f']:
    c = a + 1.
    check_type('c', c, F['a b c d e'])
    d = np.einsum('abcde,bcefg->agdef', c, b)
    try:
        check_type('d', d, F['a g e d f'])  # type annotations should be validated across arrays
    except TypeError as e:
        pass
    else:
        raise RuntimeError('Should raise a type error')
    e = np.cos(d)
    check_type('e', e, F['a g d e f'])
    return e

if __name__ == '__main__':
    a = np.ones((2, 3, 4, 5, 6))
    b = np.ones((3, 4, 6, 7, 8))
    f1(a, b)
    f2(a, b)

    c = np.ones((21, 31, 41, 51, 61))
    d = np.ones((31, 41, 61, 71, 81))
    f1(c, d)  # type annotations should not be validated across functions
    f2(c, d)

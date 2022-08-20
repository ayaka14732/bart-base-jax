from itertools import accumulate, chain, repeat
import jax
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

# testing boilerplate
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.model.fwd_linear import fwd_linear
from lib.random.wrapper import seed2key, split_key, uniform
# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
# random key management
from itertools import accumulate, chain, repeat; from operator import itemgetter
seed = 42
keys = map(itemgetter(0), accumulate(chain((split_key(seed2key(seed)),), repeat(None)), lambda acc, _: split_key(acc[1])))
rand = lambda *shape: uniform(next(keys), shape=shape)

x = rand(3, 2, 5)
kernel = rand(5, 4)
bias = rand(4)

output = fwd_linear({'kernel': kernel, 'bias': bias}, x)

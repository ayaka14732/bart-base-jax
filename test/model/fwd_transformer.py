import jax

# testing boilerplate
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.model.fwd_transformer import fwd_transformer
from lib.random.wrapper import seed2key, split_key, uniform
jax.config.update('jax_platforms', 'cpu')
# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
# random key management
from itertools import accumulate, chain, repeat; from operator import itemgetter
seed = 42
keys = map(itemgetter(0), accumulate(chain((split_key(seed2key(seed)),), repeat(None)), lambda acc, _: split_key(acc[1])))
rand = lambda *shape: uniform(next(keys), shape=shape)

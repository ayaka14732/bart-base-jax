from itertools import accumulate, chain, repeat
import jax
import jax.numpy as np
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

# testing boilerplate
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.model.fwd_embedding import fwd_embedding
# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
# random key management
seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)

vocab_size = 128
embed_size = 3

embedding = rand(vocab_size, embed_size)
x = np.array([0, 3, 15])

params = {'embedding': embedding}

output = fwd_embedding(params, x)

assert output.shape == (len(x), embed_size)

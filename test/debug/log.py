from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import contextlib
import io
import jax
import jax.numpy as np

from lib.debug.log import log, log_shape

jax.config.update('jax_platforms', 'cpu')

arr = np.zeros((10, 10))

f = io.StringIO()
with contextlib.redirect_stdout(f):
    log('bbbcUBxRLnhSxSXLNEcDAzgPMDKFnJCgoQKGUKcYdpcQLPfj')  # test `log`
    log_shape('arr', arr)  # test `log_shape`

assert f.getvalue() == '''[log.py:16] bbbcUBxRLnhSxSXLNEcDAzgPMDKFnJCgoQKGUKcYdpcQLPfj
[log.py:17] arr: (10, 10)
'''

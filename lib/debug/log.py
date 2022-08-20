from inspect import currentframe, getframeinfo
import jax.numpy as np
import os

def log(s: str) -> None:
    frame = currentframe().f_back
    lineno = frame.f_lineno
    filename = os.path.basename(getframeinfo(frame).filename)
    print(f'[{filename}:{lineno}] {s}')

def log_shape(name: str, arr: np.ndarray) -> None:
    frame = currentframe().f_back
    lineno = frame.f_lineno
    filename = os.path.basename(getframeinfo(frame).filename)
    print(f'[{filename}:{lineno}] {name}: {arr.shape}')

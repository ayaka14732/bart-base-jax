import jax.nn as nn
import jax.numpy as np

from .fwd_linear import fwd_linear

def fwd_attention(params: dict, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # params
    q_proj: dict = params['q_proj']  # linear
    k_proj: dict = params['k_proj']  # linear
    v_proj: dict = params['v_proj']  # linear
    ff: dict = params['ff']  # linear

    _, _, d_k = q_proj['kernel'].shape

    q = fwd_linear(q_proj, dst)  # bs, n_heads, dst_len, d_k
    k = fwd_linear(k_proj, src)  # bs, n_heads, src_len, d_k
    v = fwd_linear(v_proj, src)  # bs, n_heads, src_len, d_v

    qk = np.einsum('bhdk,bhsk->bhds', q, k)  # bs, n_heads, dst_len, src_len
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    # qk = np.where(mask, qk, 0)
    qkv = np.einsum('bhds,bhsv->bhdv', qk, v)  # bs, n_heads, dst_len, d_v
    output = np.einsum('bhdv,hvm->bdm', qkv, ff)  # bs, src_len. d_model

    return output

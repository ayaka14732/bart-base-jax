import jax.nn as nn
import jax.numpy as np

from ..debug.log import log_shape

def fwd_attention(params: dict, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # params
    q_proj: dict = params['q_proj']  # linear
    k_proj: dict = params['k_proj']  # linear
    v_proj: dict = params['v_proj']  # linear
    ff: dict = params['ff']  # linear

    _, _, d_k = q_proj['kernel'].shape

    log_shape("q_proj['kernel']", q_proj['kernel'])
    log_shape("k_proj['kernel']", k_proj['kernel'])
    log_shape("v_proj['kernel']", v_proj['kernel'])

    log_shape("q_proj['bias']", q_proj['bias'])
    log_shape("k_proj['bias']", k_proj['bias'])
    log_shape("v_proj['bias']", v_proj['bias'])

    q = np.einsum('bdm,mhk->bdhk', dst, q_proj['kernel'])  # bs, dst_len, n_heads, d_k
    k = np.einsum('bsm,mhk->bshk', src, k_proj['kernel'])  # bs, src_len, n_heads, d_k
    v = np.einsum('bsm,mhv->bshv', src, v_proj['kernel'])  # bs, src_len, n_heads, d_v

    if 'bias' in q_proj:
        q += q_proj['bias']  # bs, dst_len, n_heads, d_k
        k += k_proj['bias']  # bs, src_len, n_heads, d_k
        v += v_proj['bias']  # bs, src_len, n_heads, d_v

    log_shape('q', q)
    log_shape('k', k)
    log_shape('v', v)

    qk = np.einsum('bdhk,bshk->bhds', q, k)  # bs, n_heads, dst_len, src_len
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    qk = np.where(mask, qk, 0)

    log_shape('qk', qk)

    qkv = np.einsum('bhds,bshv->bdhv', qk, v)  # bs, dst_len, n_heads, d_v

    log_shape('qkv', qkv)

    output = np.einsum('bdhv,hvm->bdm', qkv, ff['kernel'])  # bs, dst_len. d_model

    if 'bias' in ff:
        output += ff['bias']  # bs, dst_len. d_model

    log_shape('output', output)

    return output

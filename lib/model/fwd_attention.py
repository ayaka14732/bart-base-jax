import jax.nn as nn
import jax.numpy as np
from jaxtyping import b as B, f as F, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

# @jaxtyped
# @typechecker
def fwd_attention(
    params: PyTree,
    src: F['bs src_len d_model'],
    dst: F['bs dst_len d_model'],
    mask: B['bs 1 dst_len src_len'],
) -> F['bs dst_len d_ff']:
    # params
    q_proj: PyTree = params['q_proj']
    k_proj: PyTree = params['k_proj']
    v_proj: PyTree = params['v_proj']
    ff: PyTree = params['ff']

    _, _, d_k = q_proj['kernel'].shape

    ### check_type("q_proj['kernel']", q_proj['kernel'], F['d_model n_heads d_k'])
    ### check_type("k_proj['kernel']", k_proj['kernel'], F['d_model n_heads d_k'])
    ### check_type("v_proj['kernel']", v_proj['kernel'], F['d_model n_heads d_v'])

    ### check_type("q_proj['bias']", q_proj['bias'], F['n_heads d_k'])
    ### check_type("k_proj['bias']", k_proj['bias'], F['n_heads d_k'])
    ### check_type("v_proj['bias']", v_proj['bias'], F['n_heads d_v'])

    q = np.einsum('bdm,mhk->bdhk', dst, q_proj['kernel'])  # bs, dst_len, n_heads, d_k
    k = np.einsum('bsm,mhk->bshk', src, k_proj['kernel'])  # bs, src_len, n_heads, d_k
    v = np.einsum('bsm,mhv->bshv', src, v_proj['kernel'])  # bs, src_len, n_heads, d_v

    ### check_type('q', q, F['bs dst_len n_heads d_k'])
    ### check_type('k', k, F['bs src_len n_heads d_k'])
    ### check_type('v', v, F['bs src_len n_heads d_v'])

    if 'bias' in q_proj:
        q += q_proj['bias']  # bs, dst_len, n_heads, d_k
        k += k_proj['bias']  # bs, src_len, n_heads, d_k
        v += v_proj['bias']  # bs, src_len, n_heads, d_v

    ### check_type('q', q, F['bs dst_len n_heads d_k'])
    ### check_type('k', k, F['bs src_len n_heads d_k'])
    ### check_type('v', v, F['bs src_len n_heads d_v'])

    qk = np.einsum('bdhk,bshk->bhds', q, k)  # bs, n_heads, dst_len, src_len
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    qk = np.where(mask, qk, 0)
    ### check_type(qk, qk, F['bs n_heads dst_len src_len'])

    qkv = np.einsum('bhds,bshv->bdhv', qk, v)  # bs, dst_len, n_heads, d_v
    ### check_type('qkv', qkv, 'bs dst_len n_heads d_v')

    output = np.einsum('bdhv,hvm->bdm', qkv, ff['kernel'])  # bs, dst_len, d_ff
    ### check_type('output', output, F['bs dst_len d_ff'])

    if 'bias' in ff:
        output += ff['bias']  # bs, dst_len, d_ff
    ### check_type('output', output, F['bs dst_len d_ff'])

    return output

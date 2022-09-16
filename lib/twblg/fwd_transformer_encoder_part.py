import jax.numpy as np
from jaxtyping import b as B, f as F, u16 as U16, PyTree, jaxtyped
from typeguard import typechecked as typechecker

from ..model.fwd_layer_norm import fwd_layer_norm
from ..model.fwd_embedding import fwd_embedding
from ..model.fwd_transformer_encoder import fwd_transformer_encoder

# @jaxtyped
# @typechecker
def fwd_transformer_encoder_part(
    params: PyTree,
    src: U16['bs src_len'],
    mask_enc: B['bs 1 src_len src_len'],
) -> F['bs dst_len d_model']:
    # params
    embedding: dict = params['embedding']  # embedding
    encoder_embed_positions: np.ndarray = params['encoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder

    _, width_enc = src.shape

    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_flax_bart.py#L718-L719
    offset = 2

    # encoder
    src = fwd_embedding(embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)

    for encoder_layer in encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    return src

from jaxtyping import Bool as B, UInt16 as U16, jaxtyped
import numpy as onp
from typeguard import typechecked as typechecker

from .device_split import device_split
from .Data import Data

@jaxtyped
@typechecker
def prepare_data_for_model(
    src: U16[onp.ndarray, 'bs src_len'],
    mask_enc_1d: B[onp.ndarray, 'bs src_len'], 
    dst: U16[onp.ndarray, 'bs dst_len'],
    mask_dec_1d: B[onp.ndarray, 'bs dst_len'],
) -> Data:
    # TODO: is this part correct?
    labels = dst

    batch_size, *_ = dst.shape

    bos_id = 2

    eoss = onp.ones((batch_size, 1), dtype=onp.uint16) * bos_id
    dst = onp.hstack((eoss, dst[:, 1:]))

    trues = onp.ones((batch_size, 1), dtype=onp.bool_)
    mask_dec_1d = onp.hstack((trues, mask_dec_1d[:, 1:]))
    # end todo

    mask_enc = onp.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = onp.tril(onp.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = onp.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    d = src, dst, mask_enc_1d, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels
    return Data(*map(device_split, d))

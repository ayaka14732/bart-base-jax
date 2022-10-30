from jaxtyping import Array, Bool as B, UInt16 as U16
from typing import NamedTuple

class Data(NamedTuple):  # used by `prepare_data_for_model`
    src: U16[Array, 'bs src_len']
    dst: U16[Array, 'bs dst_len+1']
    mask_enc_1d: B[Array, 'bs src_len']
    mask_dec_1d: B[Array, 'bs dst_len+1']
    mask_enc: B[Array, 'bs 1 src_len src_len']
    mask_dec: B[Array, 'bs 1 dst_len+1 dst_len+1']
    mask_dec_enc: B[Array, 'bs 1 dst_len+1 src_len']
    labels: U16[Array, 'bs dst_len+1']

from jaxtyping import Array
from typing import NamedTuple

class Data(NamedTuple):
    src: Array
    dst: Array
    mask_enc_1d: Array
    mask_dec_1d: Array
    mask_enc: Array
    mask_dec: Array
    mask_dec_enc: Array
    labels: Array

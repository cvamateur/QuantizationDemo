from typing import Tuple

from ..defs import *
from ..q_linear import linear_quantize


def linear_quantize_weight(weight: t_Float32Tensor,
                           bitwidth: int,
                           dim: int = 0,
                           dtype: t_dtype = t_int8) -> Tuple[t_Tensor, t_scale, t_zero_point]:
    """
    Per_tensor Symmetrical Quantization.
    """
    return linear_quantize(weight, bitwidth, Q_SYMMETRICAL, dim, dtype)


def linear_quantize_weight_per_channel(weight: t_Float32Tensor,
                                       bitwidth: int,
                                       dim: int = 0,
                                       dtype: t_dtype = t_int8) -> Tuple[t_Tensor, t_scale, t_zero_point]:
    """
    Per_channel Symmetrical Quantization.
    """
    return linear_quantize(weight, bitwidth, Q_SYMMETRICAL | Q_PER_CHANNEL, dim, dtype)

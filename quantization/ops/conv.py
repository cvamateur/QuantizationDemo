import torch
import torch.nn as nn

from ..q_types import *


def shift_quantized_bias_conv(quant_bias: t_Int32Tensor,
                              quant_weight: t_Int8Tensor,
                              input_zero_point: int) -> t_Int32Tensor:
    assert (quant_bias.dtype == t_int32), "error: bias must be of type int32"
    assert (isinstance(input_zero_point, int))
    return quant_bias - quant_weight.sum((1, 2, 3)).to(t_int32) * input_zero_point


class QuantizedConv2d(nn.Module):

    def __init__(self, weight: t_Int8Tensor,
                 bias: t_Int32Tensor,
                 input_scale: t_scale,
                 weight_scale: t_scale,
                 output_scale: t_scale,
                 input_zero_point: t_zero_point,
                 output_zero_point: t_zero_point,
                 ):
        ...
import torch
import torch.nn.functional as F

from ..q_types import *
from .other import Q_MIN, Q_MAX


def quantized_linear(input: t_Int8Tensor,
                     weight: t_Int8Tensor,
                     bias: Optional[t_Int32Tensor],
                     bitwidth_activation: int,
                     bitwidth_weight: int,
                     input_zero_point: t_zero_point,
                     output_zero_point: t_zero_point,
                     intput_scale: t_scale,
                     weight_scale: t_scale,
                     output_scale: t_scale) -> t_Int8Tensor:
    assert (input.dtype == t_int8)
    assert (weight.dtype == input.dtype)
    assert (bias is None or bias.dtype == t_int32)

    # integer based fully-connected (8-bit multiplication with 32-bit accumulation)
    if "cpu" in input.device.type:
        output = F.linear(input.to(t_int32), weight.to(t_int32), None)
    else:
        output = F.linear(input.float(), weight.float(), None)
        output = output.round().to(t_int32)

    if bias is not None:
        output = output + bias.view(1, -1)

    scale = intput_scale * weight_scale.view(1, -1) / output_scale
    output = output * scale + output_zero_point
    output = output.round().clamp(Q_MIN(bitwidth_activation), Q_MAX(bitwidth_activation))
    output = output.to(t_int8)
    return output


def shift_quantized_bias_fc(quant_bias: t_Int32Tensor,
                            quant_weight: t_Int8Tensor,
                            input_zero_point: int) -> t_Int32Tensor:
    assert (quant_bias.dtype == t_int32), "error: bias must be of type int32"
    assert (isinstance(input_zero_point, int))
    return quant_bias - quant_weight.sum(1).to(t_int32) * input_zero_point

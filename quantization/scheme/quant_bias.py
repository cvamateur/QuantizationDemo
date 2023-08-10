from typing import Tuple

from ..defs import *
from ..q_linear import linear_quantize_asymmetric, linear_quantize


def linear_quantize_bias(bias: t_Float32Tensor,
                         weight_scale: t_scale,
                         input_scale: t_scale,
                         bitwidth: int = 32,
                         dtype: t_dtype = t_int32) -> Tuple[t_Tensor, t_scale, int]:
    """
    Symmetrical Quantization. Usually weight_scale is tensor and input_scale is a scalar,
    in such case, the function performs a symmetrical per_channel quantization.

    The scale of bias is set to S_input * S_weight, which is required by hardware.
    Same as:
        linear_quantize(bias, bitwidth, Q_SYMMETRICAL, bias_scale, 0)[0]
    """
    assert (bias.dim() == 1 and bias.dtype == t_float32)
    if isinstance(input_scale, t_Tensor):
        assert (input_scale.dtype == t_float32)
        assert (input_scale.numel() == bias.numel())
        input_scale = input_scale.view(-1)
    if isinstance(weight_scale, t_Tensor):
        assert (weight_scale.dtype == t_float32)
        assert (weight_scale.numel() == bias.numel())
        weight_scale = weight_scale.view(-1)

    bias_scale = weight_scale * input_scale
    qc = make_policy(bitwidth, Q_SYMMETRICAL)
    quantized_bias = linear_quantize_asymmetric(bias, bias_scale, 0, qc, dtype)
    return quantized_bias, bias_scale, 0





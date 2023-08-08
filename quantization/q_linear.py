from typing import Union, Optional, Tuple

from .defs import *


def get_quantization_constants(tensor: t_Float32Tensor,
                               qc: QuantConfig,
                               r_min: Optional[float] = None,
                               r_max: Optional[float] = None) -> Tuple[float, int]:
    """
    Get quantization scale and zero_point for single tensor.
    """
    if r_min is None or r_max is None:
        r_min, r_max = qc.range_fn(tensor, qc.symmetric)
    q_min, q_max = qc.q_min, qc.q_max
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - r_min / scale
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = qc.round_fn(zero_point)
    return scale, int(zero_point)


def get_quantization_constants_per_channel(tensor: t_Float32Tensor,
                                           qc: QuantConfig) -> Tuple[t_Float32Tensor, t_Int32Tensor]:
    num_channels = tensor.shape[qc.channel_dim]
    scale = tensor.new_zeros([num_channels])
    zero_point = tensor.new_zeros([num_channels], dtype=t_int32)
    for i in range(num_channels):
        subtensor_i = tensor.select(qc.channel_dim, i)
        scale_i, zero_point_i = get_quantization_constants(subtensor_i, qc)
        scale[i] = scale_i
        zero_point[i] = zero_point_i
    scale_shape = [1] * tensor.dim()
    scale_shape[qc.channel_dim] = -1
    scale = scale.view(scale_shape)
    zero_point = zero_point.view(scale_shape)
    zero_point = 0 if qc.symmetric else zero_point
    return scale, zero_point


def linear_dequantize(tensor: Union[t_Int8Tensor, t_Int32Tensor],
                      scale: t_scale,
                      zero_point: t_zero_point) -> t_Float32Tensor:
    assert (
            tensor.dtype in (t_int8, t_int32, t_int16)
    ), f"error: a int tensor is required, got {tensor.dtype}"
    assert (
            isinstance(scale, float) or
            scale.dtype == t_float32 and
            scale.dim() == tensor.dim()
    ), "error: scale must be scalar or tensor of type float32"
    assert (
            isinstance(zero_point, int) or
            zero_point.dtype == t_int32 and
            zero_point.dim() == tensor.dim()
    ), f"error: zero_point must be a scalar or tensor of type {t_int32}"

    shifted_tensor = tensor - zero_point
    scaled_tensor = shifted_tensor.float() * scale
    return scaled_tensor


def linear_quantize_asymmetric(tensor: t_Float32Tensor,
                               scale: t_scale,
                               zero_point: t_zero_point,
                               qc: QuantConfig,
                               dtype: t_dtype = t_int8) -> t_Tensor:
    assert (
            tensor.dtype == t_float32
    ), f"error: a float32 tensor is required, got {tensor.dtype}"
    assert (
            isinstance(scale, float) or
            scale.dtype == t_float32 and
            scale.dim() == tensor.dim()
    ), "error: scale must be scalar or tensor of type float32"
    assert (
            isinstance(zero_point, int) or
            zero_point.dtype == t_int32 and
            zero_point.dim() == tensor.dim()
    ), f"error: zero_point must be a scalar or tensor of type {t_int32}"

    scaled_tensor = qc.scale_fn(tensor, scale)
    rounded_tensor = qc.round_fn(scaled_tensor).to(t_int32)
    shifted_tensor = rounded_tensor + zero_point
    quantized_tensor = shifted_tensor.clamp_(qc.q_min, qc.q_max)
    return quantized_tensor.to(dtype)


def linear_quantize(tensor: t_Float32Tensor,
                    bitwidth: int,
                    policy: int,
                    scale: Optional[t_scale] = None,
                    zero_point: Optional[t_zero_point] = None,
                    dim: Optional[int] = None,
                    dtype: t_dtype = t_int8) -> Tuple[t_Tensor, t_scale, t_zero_point]:
    """
    Linear quantize tensor, the function currently supports the all combination
    of the following policies:
        SYMMETRICAL  vs.  ASYMMETRICAL
        PER-TENSOR   vs.  PER-CHANNEL
        Rounding methods:
            half-to-even
            half-away-from-zero
        Ranging methods:
            absolute minmax
            percentile(99)

    @Params:
        tensor: A single to be quantized
        bitwidth: Quantization bitwidth
        policy:
            Determines how to quantize, use bitwise or operator `|` to combine policies.
            e.g.:
                Symmetrical per-tensor quantization using absolute ranging and
                half-to-even rounding method:
                    SYMMETRICAL | PER_TENSOR | RANGE_ABSOLUTE | ROUND_HALF_TO_EVEN

        scale: Optional, if None, scale will be calculated on the fly.
        zero_point: Optional, if None, zero_point will be calculated on the fly.
        dim: Optional, only useful when PER_TENSOR presents in policy.
        dtype: Data type of the output quantized tensor.

    @Return
        A tuple of (quantized tensor, scale, zero_pointer)
    """
    qc = make_policy(bitwidth, policy, dim)

    if scale is None and zero_point is None:
        if qc.per_channel:
            scale, zero_point = get_quantization_constants_per_channel(tensor, qc)
        else:
            scale, zero_point = get_quantization_constants(tensor, qc)
    assert scale is not None and zero_point is not None

    quantized_tensor = linear_quantize_asymmetric(tensor, scale, zero_point, qc, dtype)
    return quantized_tensor, scale, zero_point

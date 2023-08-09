import torch
import torch.nn as nn

from ..q_types import *
from ..functional import quantized_linear


class QuantizedLinear(nn.Module):

    def __init__(self, weight: t_Int8Tensor,
                 bias: Optional[t_Int32Tensor],
                 bitwidth_x: int,
                 bitwidth_w: int,
                 input_zero_point: t_zero_point,
                 output_zero_point: t_zero_point,
                 intput_scale: t_scale,
                 weight_scale: t_scale,
                 output_scale: t_scale):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = intput_scale
        self.weight_scale = weight_scale
        self.output_scale = output_scale

        self.bitwidth_x = bitwidth_x
        self.bitwidth_w = bitwidth_w

    def forward(self, x: t_Int8Tensor):
        return quantized_linear(
            x, self.weigth, self.bias, self.bitwidth_x, self.bitwidth_w,
            self.input_zero_point, self.output_zero_point, self.input_scale,
            self.weight_scale, self.output_scale)

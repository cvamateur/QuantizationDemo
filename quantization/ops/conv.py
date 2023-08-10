import torch
import torch.nn as nn

from ..q_types import *
from ..functional import quantized_conv2d




class QuantizedConv2d(nn.Module):

    def __init__(self, weight: t_Int8Tensor,
                 bias: t_Int32Tensor,
                 input_scale: t_scale,
                 weight_scale: t_scale,
                 output_scale: t_scale,
                 input_zero_point: t_zero_point,
                 output_zero_point: t_zero_point,
                 stride: Union[int, Tuple[int, ...]],
                 padding: Union[int, Tuple[int,...]],
                 dilation: Tuple[int, int] = 1,
                 groups: int = 1,
                 bitwidth_x: int = 8,
                 bitwidth_w: int = 8):

        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.bitwidth_x = bitwidth_x
        self.bitwidth_w = bitwidth_w

    def forward(self, x: t_Int8Tensor):
        return quantized_conv2d(
            x, self.weight, self.bias, self.bitwidth_x, self.bitwidth_w,
            self.input_zero_point, self.output_zero_point, self.input_scale,
            self.weight_scale, self.output_scale, self.stride, self.padding,
            self.dilation, self.groups,
        )

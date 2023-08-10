import torch
import torch.nn as nn

from ..q_types import t_Int8Tensor, t_int8


class QuantizedMaxPool2d(nn.MaxPool2d):

    def forward(self, x: t_Int8Tensor):
        return super().forward(x.float()).to(t_int8)


class QuantizedAvgPool2d(nn.AvgPool2d):

    def forward(self, x: t_Int8Tensor):
        return super().forward(x.float()).to(t_int8)


class QuantizedAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def forward(self, x: t_Int8Tensor):
        return super().forward(x.float()).to(t_int8)

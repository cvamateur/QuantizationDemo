import torch
import torch.nn as nn


def Q_MIN(b: int, sign: int = True) -> int:
    return -(1 << (b - 1)) if sign else 0


def Q_MAX(b: int, sign: int = True) -> int:
    return ((1 << (b - 1)) - 1) if sign else ((1 << b) - 1)


def fuse_conv_bn(conv, bn):
    # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

    return conv
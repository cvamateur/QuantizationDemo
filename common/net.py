from typing import Tuple, Union, List
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_MODULES = (
    16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128,
)


class ConvReLU(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int,
                 padding: int,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int,
                 padding: int,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class MNIST_Net(nn.Module):

    def __init__(self, factor: float = 1.0,
                 in_channels: int = 1,
                 block: nn.Module = ConvReLU,
                 plan: Tuple[Union[int, str], ...] = _DEFAULT_MODULES):
        super().__init__()

        feature_extractor = []
        for out_channels in plan:
            if isinstance(out_channels, str) and out_channels == 'M':
                feature_extractor.append(nn.MaxPool2d(3, 2, 1))
            elif isinstance(out_channels, int):
                out_channels = int(round(factor * out_channels))
                feature_extractor.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                feature_extractor.append(nn.ReLU(inplace=True))
                # feature_extractor.append(block(in_channels, out_channels, 3, 1, 1))
                in_channels = out_channels
            else:
                msg = f"error: modules plan got unknown parameter: {out_channels}"
                raise ValueError(msg)
        feature_extractor.append(nn.AdaptiveAvgPool2d(1))

        self.backbone = nn.Sequential(*feature_extractor)
        self.classifier = nn.Linear(in_channels, 10)

        self._init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))
    add("avgpool", nn.AvgPool2d(2))
    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    # x = x.mean([2, 3])
    x = x.view(x.shape[0], -1)

    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x


if __name__ == '__main__':
    from torchsummary import summary

    net = MNIST_Net(factor=1.0)

    summary(net, torch.randn([1, 1, 28, 28], device="cpu"))

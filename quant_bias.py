import time

import torch
import torch.nn as nn
from common.net import MNIST_Net, VGG
from common.dataloader import get_mnist_dataloader, get_cifar10_dataset
from common.visualize import plot_weight_distribution
from common.cli import get_parser

import quantization as q

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

vgg_path = r"./ckpt/vgg.cifar.pretrained.pth"
ckpt_path = r"./ckpt/best.pth"

USE_VGG = True


def fuse_conv_bn(conv, bn):
    # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

    return conv

def main(args):
    criterion = torch.nn.CrossEntropyLoss()
    if not USE_VGG:
        model = MNIST_Net(args.factor)
        model.load_state_dict(torch.load(ckpt_path))
        ds_train, ds_valid = get_mnist_dataloader(args)
    else:
        model = VGG()
        model.load_state_dict(torch.load(vgg_path)["state_dict"])
        fused_backbone = []
        ptr = 0
        while ptr < len(model.backbone):
            if isinstance(model.backbone[ptr], nn.Conv2d) and \
                    isinstance(model.backbone[ptr + 1], nn.BatchNorm2d):
                fused_backbone.append(fuse_conv_bn(
                    model.backbone[ptr], model.backbone[ptr + 1]))
                ptr += 2
            else:
                fused_backbone.append(model.backbone[ptr])
                ptr += 1
        model.backbone = nn.Sequential(*fused_backbone)
        ds_train, ds_valid = get_cifar10_dataset(args)

    # -------------------------------------------------
    policy_weight = q.Q_SYMMETRICAL  | q.Q_PER_CHANNEL | q.RANGE_ABSOLUTE
    policy_activation = q.Q_ASYMMETRICAL | q.RANGE_QUANTILE
    policy_bias   = q.Q_SYMMETRICAL  | q.RANGE_ABSOLUTE

    bitwidth = 4

    input_stats, output_stats = q.calibrate_activations(model, ds_valid, policy_bias, 10)

    count = 1
    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            qw, sw, zw = q.linear_quantize(m.weight, bitwidth, policy_weight, dim=0)

            qc = q.make_policy(bitwidth, policy_bias)

            # input scale and zero_point
            r_min_in = input_stats[name]["min"]
            r_max_in = input_stats[name]["max"]
            input_scale, input_zero_point = q.get_quantization_constants(qc, r_min_in, r_max_in)

            # output scale and zero_point
            r_min_out = output_stats[name]["min"]
            r_max_out = output_stats[name]["max"]
            output_scale, output_zero_point = q.get_quantization_constants(qc, r_min_out, r_max_out)

            q.linear_quantize_bias(m.bias, s, 1.0)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
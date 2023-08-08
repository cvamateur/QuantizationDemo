import time

import torch

from common.net import MNIST_Net, VGG
from common.dataloader import get_mnist_dataloader
from common.visualize import plot_weight_distribution
from common.cli import get_parser

import quantization as q

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

vgg_path = r"./ckpt/vgg.cifar.pretrained.pth"
ckpt_path = r"./ckpt/best.pth"

USE_VGG = True

def main(args):
    ds_train, ds_valid = get_mnist_dataloader(args)
    criterion = torch.nn.CrossEntropyLoss()

    if not USE_VGG:
        model = MNIST_Net(args.factor)
        model.load_state_dict(torch.load(ckpt_path))
    else:
        model = VGG()
        model.load_state_dict(torch.load(vgg_path)["state_dict"])

    # -------------------------------------------------
    weight_policy = q.Q_ASYMMETRICAL | q.Q_PER_CHANNEL | q.RANGE_ABSOLUTE
    weight_policy = q.Q_SYMMETRICAL  | q.Q_PER_CHANNEL | q.RANGE_ABSOLUTE
    bitwidth = 4

    count = 1
    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            plot_weight_distribution(m.weight, name, 32)
            qw, s, z = q.linear_quantize(m.weight, bitwidth, weight_policy, dim=0)
            # qw, s, z = q.linear_quantize_weight_per_channel(m.weight, bitwidth, dim=0)
            plot_weight_distribution(qw, name, bitwidth)



if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
import torch

from common.net import MNIST_Net, VGG
from common.dataloader import get_mnist_dataloader, get_cifar10_dataset
from common.cli import get_parser

import quantization as q

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

vgg_path = r"./ckpt/vgg.cifar.pretrained.pth"
ckpt_path = r"./ckpt/best.pth"

USE_VGG = True

def main(args):
    criterion = torch.nn.CrossEntropyLoss()
    if not USE_VGG:
        model = MNIST_Net(args.factor)
        model.load_state_dict(torch.load(ckpt_path))
        ds_train, ds_valid = get_mnist_dataloader(args)
    else:
        model = VGG()
        model.load_state_dict(torch.load(vgg_path)["state_dict"])
        ds_train, ds_valid = get_cifar10_dataset(args)

    # -------------------------------------------------
    policy_activation = q.RANGE_ABSOLUTE
    policy_activation = q.RANGE_QUANTILE
    bitwidth: int = 8

    stats_inp, stats_out = q.calibrate_activations(model, ds_valid, policy_activation)

    from pprint import pprint
    pprint(stats_inp)
    pprint(stats_out)

    q.dump_stats(stats_inp, "Output/vgg/stats_inputs.json", indent=2)
    q.dump_stats(stats_out, "Output/vgg/stats_outputs.json", indent=2)



if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
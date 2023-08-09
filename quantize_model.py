import copy
import os.path

import torch
import torch.nn as nn

from tqdm import tqdm

from common.net import MNIST_Net, VGG
from common.dataloader import get_mnist_dataloader, get_cifar10_dataset
from common.cli import get_parser

import quantization as q
import quantization.functional as qf

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

vgg_path = r"./ckpt/vgg.cifar.pretrained.pth"
ckpt_path = r"./ckpt/best.pth"


def quantize_model(model: q.t_Module, calib_data) -> q.t_Module:
    policy_weight = q.Q_SYMMETRICAL | q.Q_PER_CHANNEL | q.RANGE_ABSOLUTE
    policy_activation = q.Q_ASYMMETRICAL | q.RANGE_QUANTILE
    policy_bias = q.Q_SYMMETRICAL | q.RANGE_ABSOLUTE
    bitwidth_activation = 8
    bitwidth_weight = 8
    bitwidth_bias = 32

    qc_x = q.make_policy(bitwidth_activation, policy_activation)
    qc_w = q.make_policy(bitwidth_weight, policy_weight)
    qc_b = q.make_policy(bitwidth_bias, policy_bias)

    # Calibrate activations
    input_stats_path =  r"./out/stats_inputs.json"
    output_stats_path = r"./out/stats_outputs.json"
    if not os.path.isfile(input_stats_path) or not os.path.isfile(output_stats_path):
        input_stats, output_stats = q.calibrate_activations(model, calib_data, policy_bias, 10)
        q.dump_stats(input_stats, input_stats_path, indent=2)
        q.dump_stats(output_stats, output_stats_path, indent=2)
    else:
        input_stats = q.load_stats(input_stats_path)
        output_stats = q.load_stats(output_stats_path)

    quantized_model = copy.deepcopy(model)
    quantized_backbone = []
    ptr = 0
    while ptr < len(quantized_model.backbone):
        if (isinstance(quantized_model.backbone[ptr], nn.Conv2d) and
                isinstance(quantized_model.backbone[ptr + 1], nn.ReLU)):
            conv = quantized_model.backbone[ptr]
            conv_name = f"backbone.{ptr}"
            relu = quantized_model.backbone[ptr + 1]
            relu_name = f"backbone.{ptr + 1}"

            input_scale, input_zero_point = q.get_quantization_constants(
                qc_x, input_stats[conv_name]["min"], input_stats[conv_name]["max"])
            output_scale, output_zero_point = q.get_quantization_constants(
                qc_x, output_stats[relu_name]["min"], output_stats[relu_name]["max"])

            quantized_w, w_scale, w_zero_point, qc_w = q.linear_quantize(
                conv.weight, bitwidth_weight, policy_weight, dim=0)
            b_scale = (w_scale * input_scale).view(-1)
            quantized_b, b_scale, b_zero_point, qc_b = q.linear_quantize(
                conv.bias, bitwidth_bias, policy_bias, b_scale, 0, dtype=q.t_int32)
            shifted_b = qf.shift_quantized_bias_conv(quantized_b, quantized_w, input_zero_point)

            quantized_conv = q.QuantizedConv2d(
                quantized_w, shifted_b, input_scale, w_scale, output_scale,
                input_zero_point, output_zero_point, conv.stride, conv.padding,
                conv.dilation, conv.groups, bitwidth_activation, bitwidth_weight)
            quantized_backbone.append(quantized_conv)
            ptr += 2

        elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):

            quantized_backbone.append(q.QuantizedMaxPool2d(
                quantized_model.backbone[ptr].kernel_size,
                quantized_model.backbone[ptr].stride))
            ptr += 1

        elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
            quantized_backbone.append(q.QuantizedAvgPool2d(
                quantized_model.backbone[ptr].kernel_size,
                quantized_model.backbone[ptr].stride))
            ptr += 1

        else:
            raise NotImplementedError

    quantized_model.backbone = nn.Sequential(*quantized_backbone)

    fc_name = "classifier"
    fc = model.classifier
    input_scale, input_zero_point = q.get_quantization_constants(
        qc_x, input_stats[fc_name]["min"], input_stats[fc_name]["max"])

    output_scale, output_zero_point = q.get_quantization_constants(
        qc_x, output_stats[fc_name]["min"], output_stats[fc_name]["max"])
    quantized_w, w_scale, w_zero_point, qc_w = q.linear_quantize(
        fc.weight, bitwidth_weight, policy_weight, dim=0)

    b_scale = (w_scale * input_scale).view(-1)
    quantized_b, b_scale, b_zero_point, qc_b = q.linear_quantize(
        fc.bias, bitwidth_bias, policy_bias, b_scale, 0, dtype=q.t_int32)
    shifted_b = q.shift_quantized_bias_fc(quantized_b, quantized_w, input_zero_point)

    quantized_model.classifier = q.QuantizedLinear(
        quantized_w, shifted_b, bitwidth_activation, bitwidth_weight,
        input_zero_point, output_zero_point, input_scale, w_scale, output_scale)

    return quantized_model


@torch.inference_mode()
def evaluate(model, dataloader):

    model.eval()

    num_samples = 0
    num_correct = 0

    for images, labels in tqdm(dataloader, desc="eval", leave=False):
        # Move the data from CPU to GPU
        images = images.cuda()
        targets = labels.cuda()

        # Quantize inputs
        quantized_images = q.linear_quantize(images, 8, q.Q_ASYMMETRICAL)[0]

        # Inference
        outputs = model(quantized_images)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()



def main(args):
    model = VGG()
    model.load_state_dict(torch.load(vgg_path)["state_dict"])
    print('Before conv-bn fusion: backbone length', len(model.backbone))
    fused_backbone = []
    ptr = 0
    while ptr < len(model.backbone):
        if isinstance(model.backbone[ptr], nn.Conv2d) and \
                isinstance(model.backbone[ptr + 1], nn.BatchNorm2d):
            fused_backbone.append(qf.fuse_conv_bn(
                model.backbone[ptr], model.backbone[ptr + 1]))
            ptr += 2
        else:
            fused_backbone.append(model.backbone[ptr])
            ptr += 1
    model.backbone = nn.Sequential(*fused_backbone)
    ds_train, ds_valid = get_cifar10_dataset(args)
    print('After conv-bn fusion: backbone length', len(model.backbone))

    # -------------------------------------------------
    quantized_model = quantize_model(model, ds_valid)
    quantized_model = quantized_model.cuda()

    acc = evaluate(quantized_model, ds_valid)
    print(f"int8 model has accuracy={acc:.2f}%")


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

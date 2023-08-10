from torchprofile import profile_macs

from ..q_types import t_Tensor, t_Module

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_model_flops(model: t_Module, inputs: t_Tensor) -> int:
    num_macs = profile_macs(model, inputs)
    return num_macs


def get_model_size(model: t_Module, data_width: int = 32) -> int:
    """
    Calculate the model size in bits.
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


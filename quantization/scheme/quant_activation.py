import functools
import torch
import itertools
import torch.nn as nn
from torch.utils.data import DataLoader

from collections import defaultdict
from typing import Dict, Tuple, Optional, Callable
from tqdm import tqdm

from ..q_types import t_Float32Tensor, t_range_fn
from ..q_policy import Q_SYMMETRICAL, DECODE_RANGING
from ..basic_funcs import RANGE_REGISTER

t_stats = Dict[str, Dict[str, float]]


def _update_stats(t: t_Float32Tensor,
                  name: str,
                  range_fn: t_range_fn,
                  stats: Dict[str, Dict[str, float]]):
    min_val, max_val = range_fn(t, False)

    # later will take the mean over these values to
    # overcome outlier effects
    stats[name]["min"] += min_val
    stats[name]["max"] += max_val


@torch.inference_mode()
def calibrate_activation_stats(model: nn.Module,
                               data_loader: DataLoader,
                               policy: int,
                               max_batches: Optional[int] = None,
                               max_samples: Optional[int] = None,
                               device: str = "cuda") -> Tuple[t_stats, t_stats]:
    """
    Return activation stats which is a dict mapping layer names to
    its input and output stats:

    @Params:
        model: Model instance of torch.nn.Module.
        data_loader: Data source for calibration.
        policy: Same argument as linear_quantize takes.
        max_batches/max_samples: Calibration will stop after these quantities are reached.

    @Return:
        A tuple of (input_stats_dict, output_stats_dict), each of them has same entry:
            input_stats[module_name]["min"]: The minimum fp32 value in layer named `module_name`.
            input_stats[module_name]["max"]: The maximum fp32 value in layer named `module_name`.
            output_stats[module_name]["min"]: Same as above.
            output_stats[module_name]["max"]: Same as above.
    """
    input_stats = defaultdict(lambda: defaultdict(float))
    output_stats = defaultdict(lambda: defaultdict(float))
    range_fn = RANGE_REGISTER.get(policy & DECODE_RANGING)

    def _record_range(module, inputs, outputs, module_name):
        inputs = inputs[0]  # inputs is a tuple of length 1
        _update_stats(inputs, module_name, range_fn, input_stats)
        _update_stats(outputs, module_name, range_fn, output_stats)

    # TODO: Add more type of module
    modules_to_record = (nn.Conv2d, nn.Linear, nn.ReLU, nn.LeakyReLU)

    all_hooks = []
    for name, module in model.named_modules():
        if isinstance(module, modules_to_record):
            all_hooks.append(module.register_forward_hook(
                functools.partial(_record_range, module_name=name)))

    use_gpu = "cuda" in device and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    model.eval()

    count_batches = 0
    count_samples = 0
    progress_bar = tqdm(data_loader, desc="Calibration...")
    for i, (images, _) in enumerate(progress_bar):
        if use_gpu:
            images = images.to(device, non_blocking=True)

        # forward pass to update stats
        model(images)

        count_batches += 1
        count_samples += images.shape[0]
        if max_batches is not None and count_batches >= max_batches:
            break
        if max_samples is not None and count_samples >= max_samples:
            break

    for h in all_hooks:
        h.remove()

    for stats in itertools.chain(
            input_stats.values(), output_stats.values()):
        stats["min"] /= count_batches
        stats["max"] /= count_batches

    return input_stats, output_stats

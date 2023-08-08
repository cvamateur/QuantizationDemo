import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict, Tuple



def calibrate_activation_stats(model: nn.Module,
                               data_loader: DataLoader,
                               policy: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return calibrated result which is a dict mapping layer names to
    its input and output stats.

    policy: either RANGE_ABSOLUTE or RANGE_QUANTILE
    """

    def _forward_hook_record_range(self, x, y, module_name):
        ...

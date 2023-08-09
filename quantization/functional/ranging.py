from typing import Tuple

import numpy as np

from ..q_policy import (
    PolicyRegister,
    RANGE_ABSOLUTE,
    RANGE_QUANTILE,
    RANGE_KL_DIVERGENCE,
    RANGE_ACIQ,
)

from ..q_types import t_Float32Tensor

RANGE_REGISTER = PolicyRegister("Range")


@RANGE_REGISTER(RANGE_ABSOLUTE)
def range_absolute_minmax(t: t_Float32Tensor, symmetrical: bool = False) -> Tuple[float, float]:
    """
    Absolute range policy:
        r_min = t.min()
        r_max = t.max()
    """
    r_min = t.min().item()
    r_max = t.max().item()
    if symmetrical:
        r_max = max(abs(r_min), abs(r_max), 5e-7)
        r_min = -r_max

    return r_min, r_max


@RANGE_REGISTER(RANGE_QUANTILE)
def range_quantile_max(t: t_Float32Tensor, symmetrical: bool = True) -> Tuple[float, float]:
    """
    Quantile range policy:
        r_min = -r_max
        r_max = t.abs().quantile(99)
    """
    t = t.view(-1).abs().detach().cpu().numpy()
    r_max = np.percentile(t, 99)
    r_min = -r_max
    return r_min, r_max


@RANGE_REGISTER(RANGE_KL_DIVERGENCE)
def range_kl_divergence(t: t_Float32Tensor, symmetrical: bool = True) -> Tuple[float, float]:
    raise NotImplementedError


@RANGE_REGISTER(RANGE_ACIQ)
def range_aciq(t: t_Float32Tensor, symmetrical: bool = False) -> Tuple[float, float]:
    raise NotImplementedError

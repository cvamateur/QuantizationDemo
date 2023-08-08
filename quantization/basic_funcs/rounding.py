from typing import Union, Optional

import torch

from ..q_policy import (
    PolicyRegister,
    ROUND_HALF_TO_EVEN,
    ROUND_HALF_AWAY_FROM_ZERO,
)
from ..q_types import (
    t_Float32Tensor,
    t_Int32Tensor,
    t_float32,
)



ROUND_REGISTER = PolicyRegister("Rounding")


@ROUND_REGISTER(ROUND_HALF_TO_EVEN)
def _round_half_to_even(t: Union[float, t_Float32Tensor],
                        out: Optional[t_Int32Tensor] = None) -> t_Int32Tensor:
    if isinstance(t, float):
        t = torch.tensor(t, dtype=t_float32)
        t = torch.round(t, out=out)
        return t.item()
    else:
        return torch.round(t, out=out)



@ROUND_REGISTER(ROUND_HALF_AWAY_FROM_ZERO)
def _round_half_away_from_zero(t: Union[float, t_Float32Tensor],
                               out: Optional[t_Int32Tensor] = None) -> t_Int32Tensor:
    if isinstance(t, float):
        t = torch.tensor(t, dtype=t_float32)
        t = torch.where(t >= 0, torch.floor(t + 0.5), torch.ceil(t - 0.5), out=out)
        return t.item()
    else:
        return torch.where(t >= 0, torch.floor(t + 0.5), torch.ceil(t - 0.5), out=out)


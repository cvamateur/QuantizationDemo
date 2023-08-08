from typing import NamedTuple, Callable

from .q_types import t_scale_fn
from .q_types import t_round_fn
from .q_types import t_range_fn

#############################
#### Quantization Policy ####
#############################
Q_ASYMMETRICAL = 0
Q_SYMMETRICAL = 1
Q_LINEAR = 0
Q_NO_LINEAR = 2  # not supported
Q_PER_TENSOR = 0
Q_PER_CHANNEL = 4
Q_POWER_OF_TWO = 8

#########################
#### Rounding Policy ####
#########################
ROUND_HALF_TO_EVEN = 0
ROUND_HALF_AWAY_FROM_ZERO = 16      # 1 << 4
ROUND_HALF_TOWARDS_ZERO = 32        # 2 << 4
ROUND_HALF_DOWN = 48                # 3 << 4
ROUND_HALF_UP = 64                  # 4 << 4
DECODE_ROUNDING = 112               # 7 << 4

########################
#### Ranging Policy ####
########################
RANGE_ABSOLUTE = 0
RANGE_QUANTILE = 128                # 1 << 7
RANGE_KL_DIVERGENCE = 256           # 2 << 7
DECODE_RANGING = 384                # 3 << 7


class QuantConfig(NamedTuple):
    bitwidth: int
    q_min: int
    q_max: int
    symmetric: bool
    per_channel: bool
    channel_dim: int
    round_fn: t_round_fn
    range_fn: t_range_fn
    scale_fn: t_scale_fn


def make_policy(bitwidth: int, policy: int, channel_dim: int = 0):
    from quantization import Q_MIN, Q_MAX, RANGE_REGISTER, ROUND_REGISTER
    from quantization.basic_funcs import pow_of_two

    symmetrical = policy & Q_SYMMETRICAL

    return QuantConfig(
        bitwidth,
        Q_MIN(bitwidth) if not symmetrical else -Q_MAX(bitwidth),
        Q_MAX(bitwidth),
        symmetrical,
        policy & Q_PER_CHANNEL,
        0 if channel_dim is None else channel_dim,
        ROUND_REGISTER.get(policy & DECODE_ROUNDING),
        RANGE_REGISTER.get(policy & DECODE_RANGING),
        pow_of_two if policy & Q_POWER_OF_TWO else lambda x, s: x / s)


class PolicyRegister(object):

    def __init__(self, name: str):
        self.name = name
        self._funcs = {}

    def __call__(self, policy: int):
        def __wrapper(func: Callable):
            self._funcs[policy] = func
            return func

        return __wrapper

    def get(self, policy: int) -> Callable:
        func = self._funcs.get(policy, None)
        if func is None:
            msg = f"error: {self.name} policy ({policy}) not implemented yet!"
            raise NotImplementedError(msg)
        return func

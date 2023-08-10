from typing import NamedTuple, Callable

from .q_types import t_scale_fn
from .q_types import t_round_fn
from .q_types import t_range_fn

"""
policy is a int32 number where following bits are used:

    index   num bits    meaning
        0       1       asymmetrical(0) / symmetrical(1)
        1       1       linear(0) / non-linear(1)
        2       1       per-tensor(0) / per-channel(1)
        3       1       normal(0) / pow-of-two(1)
      4-7       4       rounding policy
     8-11       4       ranging policy
       15       1       signed(0) / unsigned(1)
"""

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
Q_SIGN = 0
Q_UNSIGN = 32768

#########################
#### Rounding Policy ####
#########################
ROUND_HALF_TO_EVEN = 0
ROUND_HALF_AWAY_FROM_ZERO = 16
ROUND_HALF_TOWARDS_ZERO = 32
ROUND_HALF_DOWN = 48
ROUND_HALF_UP = 64
DECODE_ROUNDING = 240

########################
#### Ranging Policy ####
########################
RANGE_ABSOLUTE = 0
RANGE_QUANTILE = 256
RANGE_KL_DIVERGENCE = 512
RANGE_ACIQ = 768
DECODE_RANGING = 3840


class QuantConfig(NamedTuple):
    bitwidth: int
    q_min: int
    q_max: int
    signed: bool
    symmetric: bool
    per_channel: bool
    channel_dim: int
    round_fn: t_round_fn
    range_fn: t_range_fn
    scale_fn: t_scale_fn


def make_policy(bitwidth: int, policy: int, channel_dim: int = 0):
    import quantization.functional as qf

    symmetrical = policy & Q_SYMMETRICAL
    signed = ((policy >> 15) & 1) == 0

    q_min = qf.Q_MIN(bitwidth, signed)
    q_max = qf.Q_MAX(bitwidth, signed)
    if symmetrical and signed:
        q_min = -q_max

    return QuantConfig(
        bitwidth,
        q_min,
        q_max,
        signed,
        symmetrical,
        policy & Q_PER_CHANNEL,
        0 if channel_dim is None else channel_dim,
        qf.ROUND_REGISTER.get(policy & DECODE_ROUNDING),
        qf.RANGE_REGISTER.get(policy & DECODE_RANGING),
        qf.pow_of_two if policy & Q_POWER_OF_TWO else lambda x, s: x / s)


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

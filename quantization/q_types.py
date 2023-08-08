import torch
from typing import Union, Optional, Callable, Any, Tuple

t_Tensor = torch.Tensor
t_Int8Tensor = Union[torch.CharTensor, torch.cuda.CharTensor]
t_Int16Tensor = Union[torch.ShortTensor, torch.cuda.ShortTensor]
t_Int32Tensor = Union[torch.IntTensor, torch.cuda.IntTensor]
t_Int64Tensor = Union[torch.LongTensor, torch.cuda.LongTensor]
t_Float32Tensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
t_Float64Tensor = Union[torch.DoubleTensor, torch.cuda.DoubleTensor]

t_dtype = torch.dtype
t_int8 = torch.int8
t_int16 = torch.int16
t_int32 = torch.int32
t_int64 = torch.int64
t_float32 = torch.float32
t_float64 = torch.float64

t_scale = Union[float, t_Float32Tensor]
t_zero_point = Union[int, t_Int32Tensor]
t_shift = int

t_scale_fn = Callable[[t_Float32Tensor, Union[float, t_Float32Tensor]], t_Float32Tensor]
t_round_fn = Callable[[t_Float32Tensor], t_Float32Tensor]
t_range_fn = Callable[[t_Float32Tensor, bool], Tuple[float, float]]
t_device = torch.device

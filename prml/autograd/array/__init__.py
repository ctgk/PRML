from prml.autograd.array.broadcast import broadcast_to
from prml.autograd.array.flatten import flatten
from prml.autograd.array.reshape import reshape, reshape_method
from prml.autograd.array.split import split
from prml.autograd.array.transpose import transpose, transpose_method
from prml.autograd.tensor.tensor import Tensor


Tensor.flatten = flatten
Tensor.reshape = reshape_method
Tensor.transpose = transpose_method

__all__ = [
    "broadcast_to",
    "flatten",
    "reshape",
    "split",
    "transpose"
]

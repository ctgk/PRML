from prml.autodiff.array.ones import ones
from prml.autodiff.array.reshape import reshape, reshape_method
from prml.autodiff.array.zeros import zeros
from prml.autodiff.core.array import Array


Array.reshape = reshape_method


__all__ = [
    "ones",
    "reshape",
    "zeros"
]

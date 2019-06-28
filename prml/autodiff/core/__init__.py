from prml.autodiff.core.array import Array, array, asarray
from prml.autodiff.core.backward import backward
from prml.autodiff.core.config import config


Array.backward = backward


__all__ = [
    "array",
    "asarray",
    "config"
]

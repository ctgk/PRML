import numpy as np

from prml.autodiff.core.array import Array
from prml.autodiff.core.config import config


def zeros(size):
    return Array(np.zeros(size, dtype=config.dtype))

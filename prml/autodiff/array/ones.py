import numpy as np

from prml.autodiff.core.array import Array
from prml.autodiff.core.config import config


def ones(size):
    return Array(np.ones(size, dtype=config.dtype))

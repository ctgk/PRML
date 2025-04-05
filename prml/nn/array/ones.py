import numpy as np

from prml.nn.array.array import Array
from prml.nn.config import config


def ones(size):
    return Array(np.ones(size, dtype=config.dtype))

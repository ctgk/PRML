import numpy as np
from prml.nn.array.array import Array
from prml.nn.config import config


def zeros(size):
    return Array(np.zeros(size, dtype=config.dtype))

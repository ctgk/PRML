from prml.nn.array.array import Array
from prml.nn.config import config
import numpy as np


def ones(size):
    return Array(np.ones(size, dtype=config.dtype))

import numpy as np
from prml.nn.array.array import asarray


def uniform(min, max, size):
    return asarray(np.random.uniform(min, max, size))

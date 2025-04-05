import numpy as np

from prml.nn.array.array import asarray


def uniform(min_, max_, size):
    return asarray(np.random.uniform(min_, max_, size))

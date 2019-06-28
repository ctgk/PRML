import numpy as np

from prml.autodiff.core.array import Array
from prml.autodiff.core.config import config


def ones(size):
    """
    return array filled with ones

    Parameters
    ----------
    size : int, tuple, list
        size of array to generate

    Returns
    -------
    Array
        array filled with ones
    """

    return Array(np.ones(size, dtype=config.dtype))

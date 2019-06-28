import numpy as np

from prml.autodiff._core._array import Array
from prml.autodiff._core._config import config


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

import numpy as np
from prml.autodiff._core._function import _Function


class _Log(_Function):

    @staticmethod
    def _forward(x):
        return np.log(x)

    @staticmethod
    def _backward(delta, x):
        return delta / x


def log(x):
    """
    element-wise natural logarithm of an array

    .. math::

        \log_e x

    Parameters
    ----------
    x : array_like
        power

    Returns
    -------
    Array
        exponent
    """
    return _Log().forward(x)

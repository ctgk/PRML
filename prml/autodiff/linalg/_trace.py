import numpy as np

from prml.autodiff._core._function import _Function


class _Trace(_Function):

    @staticmethod
    def _forward(x):
        return np.trace(x)

    @staticmethod
    def _backward(delta, x):
        dx = np.eye(*x.shape) * delta
        return dx


def trace(x):
    """
    computes trace of a matrix

    Parameters
    ----------
    x : array_like (D1, D2)
        matrix

    Returns
    -------
    Array
        trace of `x`

    Raises
    ------
    ValueError
        raises if `x` is not a matrix
    """
    if x.ndim != 2:
        raise ValueError
    return _Trace().forward(x)

import numpy as np

from prml.autodiff._core._function import _Function


class _LogDeterminant(_Function):

    @staticmethod
    def _forward(x):
        sign, output = np.linalg.slogdet(x)
        if np.any(sign < 1):
            raise ValueError("x is not positive-definite")
        return output

    @staticmethod
    def _backward(delta, x):
        dx = (delta.T * np.linalg.inv(np.swapaxes(x, -1, -2)).T).T
        return dx


def logdet(x):
    """
    computes log determinant

    Parameters
    ----------
    x : array_like (..., D, D)
        positive-definite matrix

    Returns
    -------
    Array
        log determinant of `x`
    """
    return _LogDeterminant().forward(x)

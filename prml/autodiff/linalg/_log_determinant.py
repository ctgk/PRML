import numpy as np

from prml.autodiff.linalg._function import _LinAlgFunction


class _LogDeterminant(_LinAlgFunction):

    @staticmethod
    def _forward(x):
        sign, output = np.linalg.slogdet(x)
        if np.any(sign < 1):
            raise ValueError("x is not positive-definite")
        return output

    @classmethod
    def _backward(cls, delta, x):
        dx = (delta.T * np.linalg.inv(cls._T(x)).T).T
        return dx


def logdet(x):
    r"""
    computes log determinant

    .. math::

        \ln|x|

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

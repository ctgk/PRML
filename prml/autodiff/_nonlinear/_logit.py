import numpy as np
from prml.autodiff._core._function import _Function


class _Logit(_Function):

    @staticmethod
    def _forward(x):
        return np.arctanh(2 * x - 1) * 2

    @staticmethod
    def _backward(delta, x):
        return delta / x / (1 - x)


def logit(x):
    r"""
    logit function which is an inverse of sigmoid function

    .. math:: \log({x \over 1-x})

    Parameters
    ----------
    x : array_like
        input

    Returns
    -------
    Array
        logit of an input
    """
    return _Logit().forward(x)

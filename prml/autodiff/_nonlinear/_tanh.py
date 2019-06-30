import numpy as np
from prml.autodiff._core._function import _Function


class _Tanh(_Function):

    def _forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def _backward(self, delta, x):
        dx = delta * (1 - self.out ** 2)
        return dx


def tanh(x):
    r"""
    tangent hyperbolic function

    .. math:: {e^x - e^{-x} \over e^x + e^{-x}}

    Parameters
    ----------
    x : array_like
        input

    Returns
    -------
    Array
        output
    """
    return _Tanh().forward(x)

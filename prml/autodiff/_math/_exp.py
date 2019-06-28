import numpy as np
from prml.autodiff._core._function import _Function


class _Exp(_Function):

    def _forward(self, x):
        self.output = np.exp(x)
        return self.output

    def _backward(self, delta, x):
        return delta * self.output


def exp(x):
    """
    element-wise exponential of an array

    .. math::

        e^{x}


    Parameters
    ----------
    x : array_like
        exponent

    Returns
    -------
    Array
        element-wise exponential of an array
    """
    return _Exp().forward(x)

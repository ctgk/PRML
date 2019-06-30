import numpy as np

from prml.autodiff._core._function import _Function


class _Determinant(_Function):

    def _forward(self, x):
        self.output = np.linalg.det(x)
        return self.output

    def _backward(self, delta, x):
        dx = delta * self.output * np.linalg.inv(np.swapaxes(x, -1, -2))
        return dx


def det(x):
    """
    compute determinant of an input

    Parameters
    ----------
    x : array_like (..., D, D)
        square matrix

    Returns
    -------
    Array
        determinant of `x`
    """
    return _Determinant().forward(x)

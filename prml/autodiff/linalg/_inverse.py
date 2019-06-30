import numpy as np

from prml.autodiff._core._function import _Function


class _Inverse(_Function):

    def _forward(self, x):
        self.output = np.linalg.inv(x)
        return self.output

    def _backward(self, delta, x):
        dx = -self.output.T @ delta @ self.output.T
        return dx


def inv(x):
    """
    computes inverse of input matrix

    Parameters
    ----------
    x : array_like (D, D)
        square matrix

    Returns
    -------
    Array
        inverse of `x`

    Raises
    ------
    ValueError
        raises error if `x` is not square
    """
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError
    return _Inverse().forward(x)

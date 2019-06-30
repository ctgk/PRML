import numpy as np

from prml.autodiff._core._function import _Function


class _Solve(_Function):

    def _forward(self, a, b):
        self.output = np.linalg.solve(a, b)
        return self.output

    def _backward(self, delta, a, b):
        db = np.linalg.solve(np.swapaxes(a, -1, -2), delta)
        da = -np.einsum("...ij,...kj->...ik", db, self.output)
        return da, db


def solve(a, b):
    """
    solve linear matrix equation

    Parameters
    ----------
    a : array_like (..., D, D)
        coefficient matrix
    b : array_like (..., D, K)
        dependent variable

    Returns
    -------
    Array
        solution of the equation
    """
    if a.shape[-1] != a.shape[-2] or a.shape[:-1] != b.shape[:-1]:
        raise ValueError
    return _Solve().forward(a, b)

import numpy as np

from prml.autodiff._core._function import _Function


class _Cholesky(_Function):

    def _forward(self, x):
        try:
            self.L = np.linalg.cholesky(x)
        except np.linalg.LinAlgError:
            raise ValueError(
                "cholesky decomposition only supports "
                "positive-definite matrix"
            )
        return self.L

    @staticmethod
    def _phi(x):
        return 0.5 * (np.tril(x) + np.tril(x, -1))

    def _backward(self, delta, x):
        delta_lower = np.tril(delta)
        P = self.phi(np.einsum("...ij,...ik->...jk", self.L, delta_lower))
        S = np.linalg.solve(
            np.swapaxes(self.L, -1, -2),
            np.einsum("...ij,...jk->...ik", P, np.linalg.inv(self.L))
        )
        dx = S + np.swapaxes(S, -1, -2) + np.tril(np.triu(S))
        return dx


def cholesky(x):
    r"""
    cholesky decomposition of positive-definite matrix

    .. math:: x = {\bf L}{\bf L}^{\rm T}

    Parameters
    ----------
    x : array_like (..., D, D)
        positive-definite matrix

    Returns
    -------
    Array
        cholesky decomposition
    """
    return _Cholesky().forward(x)

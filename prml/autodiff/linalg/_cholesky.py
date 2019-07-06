import numpy as np

from prml.autodiff.linalg._function import _LinAlgFunction


class _Cholesky(_LinAlgFunction):

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
        return np.tril(x) / (1 + np.eye(x.shape[-1]))

    @classmethod
    def _solve_trans(cls, a, b):
        return np.linalg.solve(cls._T(a), b)

    @classmethod
    def _conjugate_solve(cls, a, b):
        return cls._solve_trans(
            a,
            cls._T(
                cls._solve_trans(a, cls._T(b))
            )
        )

    def _backward(self, delta, x):
        S = self._conjugate_solve(
            self.L,
            self._phi(
                np.einsum("...ki,...kj->...ij", self.L, delta)
            )
        )
        return (S + self._T(S)) / 2


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

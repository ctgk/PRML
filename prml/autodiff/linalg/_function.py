import numpy as np

from prml.autodiff._core._function import _Function


class _LinAlgFunction(_Function):

    @staticmethod
    def _T(x):
        return np.swapaxes(x, -1, -2)

    @staticmethod
    def _dot(x, y):
        return np.tensordot(x, y, axes=((-1,), (-2,)))

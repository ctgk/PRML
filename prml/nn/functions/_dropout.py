import numpy as np

from prml import autodiff
from prml.autodiff._core._function import _Function


class _Dropout(_Function):

    def __init__(self, droprate: float = 0.5):
        self.droprate = droprate
        self.coef = 1 / (1 - droprate)

    def _forward(self, x, droprate: float = None):
        if droprate is None:
            droprate = self.droprate
            coef = self.coef
        else:
            droprate = droprate
            coef = 1 / (1 - droprate)
        self.mask = (np.random.rand(*x.shape) > droprate) * coef
        self.mask = self.mask.astype(autodiff.config.dtype)
        return x * self.mask

    def _backward(self, delta, x, droprate):
        return delta * self.mask


def dropout(x, droprate: float = 0.5):
    return _Dropout(dropout).forward(x)

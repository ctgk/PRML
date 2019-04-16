import numpy as np
from prml.nn.function import Function


class DropoutFunction(Function):

    def _forward(self, x, drop_ratio=.5):
        self.coef = 1 / (1 - drop_ratio)
        self.mask = (np.random.rand(*x.shape) > drop_ratio) * self.coef
        return x * self.mask

    def _backward(self, delta, x, drop_ratio):
        return delta * self.mask


def dropout(x, drop_ratio=.5):
    return DropoutFunction().forward(x, drop_ratio=drop_ratio)

import numpy as np
from prml.autodiff._core._function import _Function


class Tanh(_Function):

    def _forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def _backward(self, delta, x):
        dx = delta * (1 - self.out ** 2)
        return dx


def tanh(x):
    return Tanh().forward(x)

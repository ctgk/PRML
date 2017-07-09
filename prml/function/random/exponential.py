import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Exponential(Function):
    """
    sampling from exponential distribution
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.eps = np.random.exponential(1 / x.value, size=x.shape)
        output = x.value * self.eps
        return Tensor(output, function=self)

    def _backward(self, delta):
        dx = delta * self.eps
        self.x.backward(dx)


def exponential(x):
    """
    sampling from exponential distribution
    p(x|rate)
    = rate * exp(-rate * x)
    """
    return Exponential().forward(x)

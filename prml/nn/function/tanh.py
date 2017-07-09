import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Tanh(Function):
    """
    hyperbolic tangent function

    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.tanh(x.value)
        return Tensor(self.output, function=self)

    def _backward(self, delta):
        dx = (1 - np.square(self.output)) * delta
        self.x.backward(dx)


def tanh(x):
    """
    hyperbolic tangent function

    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return Tanh().forward(x)

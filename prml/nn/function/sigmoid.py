import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Sigmoid(Function):
    """
    logistic sigmoid function

    y = 1 / (1 + exp(-x))
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.tanh(x.value * 0.5) * 0.5 + 0.5
        return Tensor(self.output, function=self)

    def _backward(self, delta):
        dx = self.output * (1 - self.output) * delta
        self.x.backward(dx)


def sigmoid(x):
    """
    logistic sigmoid function

    y = 1 / (1 + exp(-x))
    """
    return Sigmoid().forward(x)

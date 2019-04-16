import numpy as np
from prml.nn.function import Function


class Sigmoid(Function):

    def _forward(self, x):
        self.out = np.tanh(x * 0.5) * 0.5 + 0.5
        return self.out

    def _backward(self, delta, x):
        return delta * self.out * (1 - self.out)


def sigmoid(x):
    return Sigmoid().forward(x)

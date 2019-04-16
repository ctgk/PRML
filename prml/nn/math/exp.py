import numpy as np
from prml.nn.function import Function


class Exp(Function):

    def _forward(self, x):
        self.output = np.exp(x)
        return self.output

    def _backward(self, delta, x):
        return delta * self.output


def exp(x):
    return Exp().forward(x)

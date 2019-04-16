import numpy as np
from prml.nn.function import Function


class Sqrt(Function):

    def _forward(self, x):
        self.output = np.sqrt(x)
        return self.output

    def _backward(self, delta, x):
        return 0.5 * delta / self.output


def sqrt(x):
    return Sqrt().forward(x)

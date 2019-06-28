import numpy as np
from prml.autodiff._core._function import _Function


class Sqrt(_Function):

    def _forward(self, x):
        self.output = np.sqrt(x)
        return self.output

    def _backward(self, delta, x):
        return 0.5 * delta / self.output


def sqrt(x):
    return Sqrt().forward(x)

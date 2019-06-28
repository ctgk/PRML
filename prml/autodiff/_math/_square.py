import numpy as np
from prml.autodiff._core._function import _Function


class Square(_Function):

    @staticmethod
    def _forward(x):
        return np.square(x)

    @staticmethod
    def _backward(delta, x):
        return 2 * delta * x


def square(x):
    return Square().forward(x)

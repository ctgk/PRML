import numpy as np
from prml.nn.function import Function


class Square(Function):

    @staticmethod
    def _forward(x):
        return np.square(x)

    @staticmethod
    def _backward(delta, x):
        return 2 * delta * x


def square(x):
    return Square().forward(x)

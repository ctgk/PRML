import numpy as np
from prml.nn.function import Function


class Logit(Function):

    @staticmethod
    def _forward(x):
        return np.arctanh(2 * x - 1) * 2

    @staticmethod
    def _backward(delta, x):
        return delta / x / (1 - x)


def logit(x):
    return Logit().forward(x)

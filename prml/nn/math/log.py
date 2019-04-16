import numpy as np
from prml.nn.function import Function


class Log(Function):

    @staticmethod
    def _forward(x):
        return np.log(x)

    @staticmethod
    def _backward(delta, x):
        return delta / x


def log(x):
    return Log().forward(x)

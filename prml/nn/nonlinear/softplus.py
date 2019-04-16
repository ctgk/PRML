import numpy as np
from prml.nn.function import Function


class Softplus(Function):

    @staticmethod
    def _forward(x):
        return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

    @staticmethod
    def _backward(delta, x):
        return (np.tanh(0.5 * x) * 0.5 + 0.5) * delta


def softplus(x):
    """
    y = log(1 + exp(x))
    """
    return Softplus().forward(x)

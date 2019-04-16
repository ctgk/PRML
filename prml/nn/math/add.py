import numpy as np
from prml.nn.function import Function


class Add(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x + y

    @staticmethod
    def _backward(delta, x, y):
        return delta, delta


class AddBias(Function):

    @staticmethod
    def _forward(x, y):
        return x + y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta
        dy = np.sum(delta, axis=tuple(i for i in range(x.ndim - 1)))
        return dx, dy


class AddScalar(Function):

    @staticmethod
    def _forward(x, y):
        return x + y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta
        dy = np.atleast_1d(np.sum(delta))
        return dx, dy


def add(x, y):
    return Add().forward(x, y)
    # x = Function._convert2array(x)
    # y = Function._convert2array(y)
    # if x.shape == y.shape:
    #     return Add().forward(x, y)
    # elif x.size == 1:
    #     return AddScalar().forward(y, x)
    # elif y.size == 1:
    #     return AddScalar().forward(x, y)
    # elif x.shape[-1] == y.shape[-1]:
    #     if x.ndim == 1:
    #         return AddBias().forward(y, x)
    #     elif y.ndim == 1:
    #         return AddBias().forward(x, y)
    # else:
    #     raise ValueError

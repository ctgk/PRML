import numpy as np
from prml.nn.function import Function


class Subtract(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x - y

    @staticmethod
    def _backward(delta, x, y):
        return delta, -delta


def subtract(x, y):
    return Subtract().forward(x, y)


def rsubtract(x, y):
    return Subtract().forward(y, x)

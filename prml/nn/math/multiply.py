import numpy as np
from prml.nn.function import Function


class Multiply(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x * y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta * y
        dy = delta * x
        return dx, dy


def multiply(x, y):
    return Multiply().forward(x, y)

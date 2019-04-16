import numpy as np
from prml.nn.function import Function


class Power(Function):
    """
    First array elements raised to powers from second array
    """

    def _forward(self, x, y):
        self.output = np.power(x, y)
        return self.output

    def _backward(self, delta, x, y):
        dx = y * np.power(x, y - 1) * delta
        if (x > 0).all():
            dy = self.output * np.log(x) * delta
            return dx, dy
        return dx


def power(x, y):
    """
    First array elements raised to powers from second array
    """
    return Power().forward(x, y)


def rpower(x, y):
    return Power().forward(y, x)

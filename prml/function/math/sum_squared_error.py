import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function
from prml.function.array.broadcast import broadcast_to


class SumSquredError(Function):

    def _check_input(self, x, y):
        x = self._convert2tensor(x)
        y = self._convert2tensor(y)
        if x.shape != y.shape:
            shape = np.broadcast(x.value, y.value).shape
            if x.shape != shape:
                x = broadcast_to(x, shape)
            if y.shape != shape:
                y = broadcast_to(y, shape)
        return x, y

    def _forward(self, x, y):
        x, y = self._check_input(x, y)
        self.x = x
        self.y = y
        return Tensor(0.5 * np.square(x.value - y.value).sum(), function=self)

    def _backward(self, delta):
        dx = delta * (self.x.value - self.y.value)
        dy = delta * (self.y.value - self.x.value)
        self.x.backward(dx)
        self.y.backward(dy)


def sum_squared_error(x, y):
    """
    sum of squared error

    sum(square(x - y))
    """
    return SumSquredError().forward(x, y)

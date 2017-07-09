import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Log(Function):
    """
    element-wise natural logarithm of the input

    y = log(x)
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        return Tensor(np.log(self.x.value), function=self)

    def _backward(self, delta):
        dx = delta / self.x.value
        self.x.backward(dx)


def log(x):
    """
    element-wise natural logarithm of the input

    y = log(x)
    """
    return Log().forward(x)

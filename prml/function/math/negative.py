from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Negative(Function):
    """
    element-wise negative

    y = -x
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        return Tensor(-x.value, function=self)

    def _backward(self, delta):
        dx = -delta
        self.x.backward(dx)


def negative(x):
    """
    element-wise negative
    """
    return Negative().forward(x)

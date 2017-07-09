from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Flatten(Function):
    """
    flatten array
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self._atleast_ndim(x, 2)
        self.x = x
        return Tensor(x.value.flatten(), function=self)

    def _backward(self, delta):
        dx = delta.reshape(*self.x.shape)
        self.x.backward(dx)


def flatten(x):
    """
    flatten N-dimensional array (N >= 2)
    """
    return Flatten().forward(x)

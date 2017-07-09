from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Reshape(Function):
    """
    reshape array
    """

    def _forward(self, x, shape):
        x = self._convert2tensor(x)
        self._atleast_ndim(x, 1)
        self.x = x
        return Tensor(x.value.reshape(*shape), function=self)

    def _backward(self, delta):
        dx = delta.reshape(*self.x.shape)
        self.x.backward(dx)


def reshape(x, shape):
    """
    reshape N-dimensional array (N >= 1)
    """
    return Reshape().forward(x, shape)

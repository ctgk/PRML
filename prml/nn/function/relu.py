from prml.tensor.tensor import Tensor
from prml.function.function import Function


class ReLU(Function):
    """
    Rectified Linear Unit

    y = max(x, 0)
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        return Tensor(x.value.clip(min=0), function=self)

    def _backward(self, delta):
        dx = (self.x.value > 0) * delta
        self.x.backward(dx)


def relu(x):
    """
    Rectified Linear Unit

    y = max(x, 0)
    """
    return ReLU().forward(x)

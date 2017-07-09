import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class BroadcastTo(Function):
    """
    Broadcast a tensor to an new shape
    """

    def _forward(self, x, shape):
        x = self._convert2tensor(x)
        self.x = x
        output = np.broadcast_to(x.value, shape)
        return Tensor(output, function=self)

    def _backward(self, delta):
        dx = delta
        if delta.ndim != self.x.ndim:
            dx = dx.sum(axis=tuple(range(dx.ndim - self.x.ndim)))
            if isinstance(dx, np.number):
                dx = np.array(dx)
        axis = tuple(i for i, len_ in enumerate(self.x.shape) if len_ == 1)
        if len(axis) > 0:
            dx = dx.sum(axis=axis, keepdims=True)
        self.x.backward(dx)


def broadcast_to(x, shape):
    """
    Broadcast a tensor to an new shape
    """
    return BroadcastTo().forward(x, shape)

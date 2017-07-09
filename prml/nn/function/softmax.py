import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Softmax(Function):

    def __init__(self, axis=-1):
        self.axis = axis

    def _softmax(self, array):
        y = array - np.max(array, self.axis, keepdims=True)
        np.exp(y, out=y)
        y /= y.sum(self.axis, keepdims=True)
        return y

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = self._softmax(x.value)
        return Tensor(self.output, function=self)

    def _backward(self, delta):
        dx = self.output * delta
        dx -= self.output * dx.sum(self.axis, keepdims=True)
        self.x.backward(dx)


def softmax(x, axis=-1):
    return Softmax(axis=axis).forward(x)

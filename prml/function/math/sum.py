import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Sum(Function):
    """
    summation along given axis

    y = sum_i=1^N x_i
    """

    def __init__(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, x):
        x = self._convert2tensor(x)
        self._atleast_ndim(x, 1)
        self.x = x
        output = x.value.sum(axis=self.axis, keepdims=self.keepdims)
        return Tensor(output, function=self)

    def _backward(self, delta):
        if isinstance(delta, np.ndarray) and (not self.keepdims):
            axis_positive = []
            for axis in self.axis:
                if axis < 0:
                    axis_positive.append(self.x.ndim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                delta = np.expand_dims(delta, axis)
        dx = np.broadcast_to(delta, self.x.shape)
        self.x.backward(dx)


def sum(x, axis=None, keepdims=False):
    """
    returns summation of the elements along given axis

    y = sum_i=1^N x_i
    """
    return Sum(axis=axis, keepdims=keepdims).forward(x)

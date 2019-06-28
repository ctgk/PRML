import numpy as np
from prml.autodiff._core._function import _Function


class SumAxisOrKeepdims(_Function):
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
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def _backward(self, delta, x):
        requires_expand = (
            isinstance(delta, np.ndarray)
            and (not self.keepdims) and (self.axis is not None)
        )
        if requires_expand:
            axis_positive = []
            for axis in self.axis:
                if axis < 0:
                    axis_positive.append(x.ndim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                delta = np.expand_dims(delta, axis)
        dx = np.broadcast_to(delta, x.shape)
        return dx


class SumSimple(_Function):

    @staticmethod
    def _forward(x):
        return x.sum()

    @staticmethod
    def _backward(delta, x):
        return np.broadcast_to(delta, x.shape)


def sum(x, axis=None, keepdims=False):
    """
    returns summation of the elements along given axis
    y = sum_i=1^N x_i
    """
    x = _Function._convert2array(x)
    if x.ndim == 1:
        return SumAxisOrKeepdims(axis=axis, keepdims=True).forward(x)
    elif axis is None and keepdims is False:
        return SumSimple().forward(x)
    return SumAxisOrKeepdims(axis=axis, keepdims=keepdims).forward(x)

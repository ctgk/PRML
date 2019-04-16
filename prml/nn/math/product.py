import numpy as np
from prml.nn.function import Function


class Product(Function):

    def __init__(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, tuple):
            axis = tuple(sorted(axis))
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, x):
        self.output = np.prod(x, axis=self.axis, keepdims=True)
        if not self.keepdims:
            return np.squeeze(self.output)
        else:
            return self.output

    def backward(self, delta, x):
        if not self.keepdims and self.axis is not None:
            for ax in self.axis:
                delta = np.expand_dims(delta, ax)
        dx = delta * self.output / x
        return dx


def prod(x, axis=None, keepdims=False):
    """
    product of all element in the array
    Parameters
    ----------
    x : tensor_like
        input array
    axis : int, tuple of ints
        axis or axes along which a product is performed
    keepdims : bool
        keep dimensionality or not
    Returns
    -------
    product : tensor_like
        product of all element
    """
    return Product(axis=axis, keepdims=keepdims).forward(x)

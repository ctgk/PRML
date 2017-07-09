import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class SoftmaxCrossEntropy(Function):
    """
    sum of cross entropy for one-of-k coded data
    normalization of softmax activation can be taken along arbitrary axis
    softmax activation
    y_i = exp(x_i) / sum_j{exp(x_j)}
    cross_entropy_i = -t_i * log(y_i)

    Parameters
    ----------
    axis : int
        axis to normalize softmax activation along
    x : ndarray
        input logit
    t : ndarray
        corresponding target in one-of-k coding format
    """

    def __init__(self, axis=-1):
        """
        construct softmax cross entropy function
        Parameters
        ----------
        axis : int
            axis to normalize softmax activation along
        """
        self.axis = axis

    def _check_input(self, x, t):
        x = self._convert2tensor(x)
        t = self._convert2tensor(t)
        if x.shape != t.shape:
            raise ValueError(
                "shapes {} and {} not aligned"
                .format(x.shape, t.shape)
            )
        return x, t

    def _softmax(self, array):
        y = np.exp(array - np.max(array, self.axis, keepdims=True))
        y /= np.sum(y, self.axis, keepdims=True)
        return y

    def _forward(self, x, t):
        x, t = self._check_input(x, t)
        self.x = x
        self.t = t
        self.y = self._softmax(x.value)
        np.clip(self.y, 1e-10, 1, out=self.y)
        loss = np.sum(-t.value * np.log(self.y))
        return Tensor(loss, function=self)

    def _backward(self, delta):
        dx = delta * (self.y - self.t.value)
        dt = - delta * np.log(self.y)
        self.x.backward(dx)
        self.t.backward(dt)


def softmax_cross_entropy(logit, onehot, axis=-1):
    """
    sum of cross entropy for one-of-k coded data
    """
    return SoftmaxCrossEntropy(axis).forward(logit, onehot)

import numpy as np
from scipy.special import logsumexp
from prml.autodiff._core._function import _Function


class _Softmax(_Function):

    def _forward(self, x):
        self.output = np.exp(x - logsumexp(x, axis=-1, keepdims=True))
        return self.output

    def _backward(self, delta, x):
        dx = self.output * delta
        dx -= self.output * dx.sum(axis=-1, keepdims=True)
        return dx


def softmax(x):
    r"""
    softmax activation along last axis

    .. math:: y_i = {e^{x_i} \over \sum_i e^{x_i}}

    Parameters
    ----------
    x : array_like
        input

    Returns
    -------
    Array
        output
    """
    return _Softmax().forward(x)

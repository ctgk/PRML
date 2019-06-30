import numpy as np
from scipy.special import logsumexp
from prml.autodiff._core._function import _Function


class _LogSoftmax(_Function):

    def _forward(self, x):
        self.output = x - logsumexp(x, axis=-1, keepdims=True)
        return self.output

    def _backward(self, delta, x):
        softmax = np.exp(self.output)
        dx = delta - softmax * delta.sum(axis=-1, keepdims=True)
        return dx


def log_softmax(x):
    r"""
    natural logarithm of softmax activation along last axis

    .. math:: y_i = x_i - \log\left(\sum_i \exp{x_i}\right)

    Parameters
    ----------
    x : array_like
        input

    Returns
    -------
    Array
        output
    """
    return _LogSoftmax().forward(x)

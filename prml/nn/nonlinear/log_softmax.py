import numpy as np
from scipy.special import logsumexp
from prml.nn.function import Function


class LogSoftmax(Function):

    def _forward(self, x):
        self.output = x - logsumexp(x, axis=-1, keepdims=True)
        return self.output

    def _backward(self, delta, x):
        softmax = np.exp(self.output)
        dx = delta - softmax * delta.sum(axis=-1, keepdims=True)
        return dx


def log_softmax(x):
    return LogSoftmax().forward(x)

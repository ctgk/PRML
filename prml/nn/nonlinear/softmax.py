import numpy as np
from scipy.special import logsumexp
from prml.nn.function import Function


class Softmax(Function):

    def _forward(self, x):
        self.output = np.exp(x - logsumexp(x, axis=-1, keepdims=True))
        return self.output

    def _backward(self, delta, x):
        dx = self.output * delta
        dx -= self.output * dx.sum(axis=-1, keepdims=True)
        return dx


def softmax(x):
    return Softmax().forward(x)

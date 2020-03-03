import numpy as np
from scipy.special import logsumexp
from prml.nn.function import Function


class SoftmaxCrossEntropy(Function):

    def _forward(self, x, t):
        self.log_softmax = x - logsumexp(x, axis=-1, keepdims=True)
        return -t * self.log_softmax

    def _backward(self, delta, x, t):
        dx = delta * (np.exp(self.log_softmax) - t)
        dt = -delta * self.log_softmax
        return dx, dt


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy().forward(x, t)

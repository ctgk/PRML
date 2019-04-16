import numpy as np
from prml.nn.function import Function


class SigmoidCrossEntropy(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, t):
        return np.maximum(x, 0) - t * x + np.log1p(np.exp(-np.abs(x)))

    @staticmethod
    def _backward(delta, x, t):
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        dx = delta * (y - t)
        dt = -delta * x
        return dx, dt


def sigmoid_cross_entropy(x, t):
    return SigmoidCrossEntropy().forward(x, t)

import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class SigmoidCrossEntropy(Function):
    """
    sum of cross entropies for binary data
    logistic sigmoid
    y_i = 1 / (1 + exp(-x_i))
    cross_entropy_i = -t_i * log(y_i) - (1 - t_i) * log(1 - y_i)

    Parameters
    ----------
    x : ndarary
        input logit
    y : ndarray
        corresponding target binaries
    """

    def _check_input(self, x, t):
        x = self._convert2tensor(x)
        t = self._convert2tensor(t)
        if x.shape != t.shape:
            raise ValueError(
                "shapes {} and {} not aligned"
                .format(x.shape, t.shape)
            )
        return x, t


    def _forward(self, x, t):
        x, t = self._check_input(x, t)
        self.x = x
        self.t = t
        # y = self.forward(x)
        # np.clip(y, 1e-10, 1 - 1e-10, out=y)
        # return np.sum(-t * np.log(y) - (1 - t) * np.log(1 - y))
        loss = np.sum(
            np.maximum(x.value, 0)
            - t.value * x.value
            + np.log1p(np.exp(-np.abs(x.value)))
        )
        return Tensor(loss, function=self)

    def _backward(self, delta):
        y = np.tanh(self.x.value * 0.5) * 0.5 + 0.5
        dx = delta * (y - self.t.value)
        dt = - delta * self.x.value
        self.x.backward(dx)
        self.t.backward(dt)


def sigmoid_cross_entropy(logit, label):
    """
    sum of cross entropies for binary data
    """
    return SigmoidCrossEntropy().forward(logit, label)

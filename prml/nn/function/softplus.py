import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Softplus(Function):

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        output = np.maximum(x.value, 0) + np.log1p(np.exp(-np.abs(x.value)))
        return Tensor(output, function=self)

    def _backward(self, delta):
        dx = (np.tanh(0.5 * self.x.value) * 0.5 + 0.5) * delta
        self.x.backward(dx)


def softplus(x):
    """
    smoothed rectified linear unit

    log(1 + exp(x))
    """
    return Softplus().forward(x)

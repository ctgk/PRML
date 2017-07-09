import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class Dropout(Function):

    def __init__(self, prob):
        """
        construct dropout function

        Parameters
        ----------
        prob : float
            probability of dropping the input value
        """
        if not isinstance(prob, float) or prob < 0 or prob > 1:
            raise ValueError("{} is out of the range [0, 1]".format(prob))
        self.prob = prob
        self.coef = 1 / (1 - prob)

    def _forward(self, x, istraining=False):
        x = self._convert2tensor(x)
        if istraining:
            self.x = x
            self.mask = (np.random.rand(*x.shape) > self.prob) * self.coef
            return Tensor(x.value * self.mask, function=self)
        else:
            return x

    def _backward(self, delta):
        dx = delta * self.mask
        self.x.backward(dx)


def dropout(x, prob, istraining):
    return Dropout(prob).forward(x, istraining)

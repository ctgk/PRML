import numpy as np
from .layer import Layer


class Dropout(Layer):
    """
    Dropout layer

    for each element
    y = x if random.rand() > prob else 0
    """

    def __init__(self, prob):
        """
        construct dropout layer

        Parameters
        ----------
        prob : float
            probability of dropping the input value
        """
        super().__init__()
        assert 0. <= prob < 1.
        self.prob = prob
        self.scale = 1. / (1 - prob)

    def forward(self, x, training=False):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.prob) * self.scale
            return x * self.mask
        else:
            return x

    def backward(self, delta):
        return delta * self.mask

import numpy as np

from prml.nn.initializers._initializer import Initializer


class Normal(Initializer):

    def __init__(self, mean: float, std: float):
        if std <= 0.:
            raise ValueError(
                "standard deviation of Normal initilizer must be positive")
        self.mean = mean
        self.std = std

    def _forward(self, size):
        return np.random.normal(self.mean, self.std, size)

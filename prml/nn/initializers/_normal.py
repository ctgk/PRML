import numpy as np
from scipy.stats import truncnorm

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


class TruncNormal(Normal):

    def __init__(self, mean: float, std: float, min_: float, max_: float):
        super().__init__(mean, std)
        self.a = (min_ - mean) / std
        self.b = (max_ - mean) / std

    def _forward(self, size):
        return truncnorm.rvs(
            self.a, self.b, loc=self.mean, scale=self.std, size=size)

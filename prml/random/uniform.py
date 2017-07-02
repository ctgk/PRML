import numpy as np
from prml.random.random import RandomVariable


class Uniform(RandomVariable):
    """
    Uniform distribution
    p(x|a, b)
    = 1 / ((b_0 - a_0) * (b_1 - a_1)) if a <= x <= b else 0
    """

    def __init__(self, low, high):
        """
        construct uniform distribution

        Parameters
        ----------
        low : int, float, or np.ndarray
            lower boundary
        high : int, float, or np.ndarray
            higher boundary
        """
        low = np.asarray(low)
        high = np.asarray(high)
        assert low.shape == high.shape
        assert (low <= high).all()
        self.low = low
        self.high = high
        self.value = 1 / np.prod(high - low)

    def __repr__(self):
        return "Uniform(low={0.low}, high={0.high})".format(self)

    @property
    def ndim(self):
        return self.low.ndim

    @property
    def size(self):
        return self.low.size

    @property
    def shape(self):
        return self.low.shape

    @property
    def mean(self):
        return 0.5 * (self.low + self.high)

    def _pdf(self, X):
        higher = np.logical_and.reduce(X >= self.low, 1)
        lower = np.logical_and.reduce(X <= self.high, 1)
        return self.value * np.logical_and(higher, lower)

    def _draw(self, sample_size=1):
        u01 = np.random.uniform(size=(sample_size,) + self.shape)
        return u01 * (self.high - self.low) + self.low

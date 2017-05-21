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
        low : (ndim,) np.ndarray
            lower boundary
        high : (ndim,) np.ndarray
            higher boundary
        """
        assert low.ndim == high.ndim == 1
        assert (low <= high).all()
        self.ndim = low.size
        self.low = low
        self.high = high
        self.value = 1 / np.prod(high - low)

    def __repr__(self):
        return "Uniform(low={0}, high={1})".format(self.low, self.high)

    @property
    def mean(self):
        return 0.5 * (self.low + self.high)

    def _proba(self, X):
        x1 = np.logical_and.reduce(X >= self.low, 1)
        x2 = np.logical_and.reduce(X <= self.high, 1)
        return self.value * np.logical_and(x1, x2)

    def _draw(self, sample_size=1):
        return np.random.uniform(size=(sample_size, self.ndim)) * (self.high - self.low) + self.low

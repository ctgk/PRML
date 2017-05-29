import numpy as np
from scipy.special import gamma
from prml.random.random import RandomVariable


np.seterr(all="ignore")


class Gamma(RandomVariable):
    """
    Gamma distribution
    p(x|a(shape),b(rate))
    = b^a x^(a-1) exp(-bx) / gamma(a)
    """

    def __init__(self, shape=None, rate=None):
        assert shape is None or isinstance(shape, (int, float))
        assert rate is None or isinstance(rate, (int, float))
        self.shape = shape
        self.rate = rate

    def __repr__(self):
        return "Gamma(shape={0}, rate={1})".format(self.shape, self.rate)

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(shape=self.shape, rate=self.rate / other)

    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(shape=self.shape, rate=self.rate / other)

    def __imul__(self, other):
        assert isinstance(other, (int, float))
        self.rate /= other
        return self

    def __truediv__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(shape=self.shape, rate=self.rate * other)

    def __itruediv__(self, other):
        assert isinstance(other, (int, float))
        self.rate *= other
        return self

    @property
    def mean(self):
        return self.shape / self.rate

    @property
    def var(self):
        return self.shape / self.rate ** 2

    def _pdf(self, X):
        assert np.size(X, 1) == 1
        return (
            self.rate ** self.shape
            * X ** (self.shape - 1)
            * np.exp(-self.rate * X)
            / gamma(self.shape))

    def _draw(self, sample_size=1):
        return np.random.gamma(
            shape=self.shape, scale=1 / self.rate, size=sample_size)[:, None]

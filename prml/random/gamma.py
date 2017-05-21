import numpy as np
from scipy.special import gamma
from prml.random.random import RandomVariable


np.seterr(all="ignore")


class Gamma(RandomVariable):
    """
    Gamma distribution
    p(x|a,b)
    = b^a x^(a-1) exp(-bx) / gamma(a)
    """

    def __init__(self, a=None, b=None):
        assert a is None or isinstance(a, (int, float))
        assert b is None or isinstance(b, (int, float))
        self.a = a
        self.b = b

    def __repr__(self):
        return "Gamma(a={0}, b={1})".format(self.a, self.b)

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(a=self.a, b=self.b / other)

    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(a=self.a, b=self.b / other)

    def __imul__(self, other):
        assert isinstance(other, (int, float))
        self.b /= other
        return self

    def __truediv__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(a=self.a, b=self.b * other)

    def __itruediv__(self, other):
        assert isinstance(other, (int, float))
        self.b *= other
        return self

    @property
    def mean(self):
        return self.a / self.b

    @property
    def var(self):
        return self.a / self.b ** 2

    def _proba(self, X):
        assert np.size(X, 1) == 1
        return self.b ** self.a * X ** (self.a - 1) * np.exp(-self.b * X) / gamma(self.a)

    def _draw(self, sample_size=1):
        return np.random.gamma(shape=self.a, scale=1 / self.b, size=sample_size)[:, None]

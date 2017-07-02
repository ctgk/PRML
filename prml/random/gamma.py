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

    def __init__(self, shape, rate):
        """
        construct Gamma distribution

        Parameters
        ----------
        shape : int, float, or np.ndarray
            shape parameter
        rate : int, float, or np.ndarray
            rate parameter
        """
        if isinstance(shape, (int, float)):
            shape = np.array(shape)
        if isinstance(rate, (int, float)):
            rate = np.array(rate)
        assert shape.shape == rate.shape
        self.shape_ = shape
        self.rate = rate

    def __repr__(self):
        return "Gamma(\nshape=\n{0.shape_},\nrate=\n{0.rate}\n)".format(self)

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(shape=self.shape_, rate=self.rate / other)

    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(shape=self.shape_, rate=self.rate / other)

    def __imul__(self, other):
        assert isinstance(other, (int, float))
        self.rate /= other
        return self

    def __truediv__(self, other):
        assert isinstance(other, (int, float))
        return Gamma(shape=self.shape_, rate=self.rate * other)

    def __itruediv__(self, other):
        assert isinstance(other, (int, float))
        self.rate *= other
        return self

    @property
    def ndim(self):
        return self.shape_.ndim

    @property
    def shape(self):
        return self.shape_.shape

    @property
    def size(self):
        return self.shape_.size

    @property
    def mean(self):
        return self.shape_ / self.rate

    @property
    def var(self):
        return self.shape_ / self.rate ** 2

    def _pdf(self, X):
        return (
            self.rate ** self.shape_
            * X ** (self.shape_ - 1)
            * np.exp(-self.rate * X)
            / gamma(self.shape_))

    def _draw(self, sample_size=1):
        return np.random.gamma(
            shape=self.shape_,
            scale=1 / self.rate,
            size=(sample_size,) + self.shape
        )

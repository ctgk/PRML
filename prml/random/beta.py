import numpy as np
from scipy.special import gamma
from prml.random.random import RandomVariable


np.seterr(all="ignore")


class Beta(RandomVariable):
    """
    Beta distribution
    p(mu|n_ones, n_zeros)
    = gamma(n_ones + n_zeros)
      * mu^(n_ones-1) * (1-mu)^(n_zeros-1)
      / gamma(n_ones) / gamma(n_zeros)
    """

    def __init__(self, n_ones=np.ones(1), n_zeros=np.ones(1)):
        """
        construct beta distribution

        Parameters
        ----------
        n_ones : (ndim,) np.ndarray
            pseudo count of one
        n_zeros : (ndim,) np.ndarray
            pseudo count of zero
        """
        assert isinstance(n_ones, np.ndarray)
        assert isinstance(n_zeros, np.ndarray)
        assert n_ones.ndim == n_zeros.ndim == 1
        assert n_ones.size == n_zeros.size
        self.n_ones = n_ones
        self.n_zeros = n_zeros
        self.ndim = n_ones.size

    def __repr__(self):
        return "Beta(n_ones={0.n_ones}, n_zeros={0.n_zeros})".format(self)

    @property
    def mean(self):
        return self.n_ones / (self.n_ones + self.n_zeros)

    @property
    def var(self):
        return np.diag(
            self.n_ones * self.n_zeros
            / (self.n_ones + self.n_zeros) ** 2
            / (self.n_ones + self.n_zeros + 1)
        )

    def _call(self, mu):
        return (
            gamma(self.n_ones + self.n_zeros)
            * np.power(mu, self.n_ones - 1)
            * np.power(1 - mu, self.n_zeros - 1)
            / gamma(self.n_ones)
            / gamma(self.n_zeros)
        )

    def _draw(self, sample_size=1):
        return np.random.beta(
            self.n_ones, self.n_zeros, size=(sample_size, self.ndim))

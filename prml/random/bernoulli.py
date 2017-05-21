import numpy as np
from prml.random.random import RandomVariable
from prml.random.beta import Beta


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu) = mu^x (1 - mu)^(1 - x)
    """

    def __init__(self, mu=None):
        """
        construct Bernoulli distribution

        Parameters
        ----------
        mu : (ndim,) np.ndarray or Beta
            probability of value 1
        """
        assert (mu is None or isinstance(mu, (np.ndarray, Beta)))
        self.mu = mu

    def __setattr__(self, name, value):
        if name is "mu":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert (value >= 0).all() and (value <= 1.).all()
                self.ndim = value.size
                object.__setattr__(self, name, value)
            elif isinstance(value, Beta):
                self.ndim = value.ndim
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return "Bernoulli(mu={})".format(self.mu)

    @property
    def mean(self):
        return self.mu

    @property
    def var(self):
        return np.diag(self.mu * (1 - self.mu))

    def _ml(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones, "{0} is not equal to {1} plus {2}".format(X.size, n_zeros, n_ones)
        self.mu = np.mean(X, axis=0)

    def _map(self, X):
        assert isinstance(self.mu, Beta)
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        n_ones += self.mu.n_ones
        n_zeros += self.mu.n_zeros
        self.mu = (n_ones - 1) / (n_ones + n_zeros - 2)

    def _bayes(self, X):
        assert isinstance(self.mu, Beta)
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        self.mu = Beta(
            n_ones=self.mu.n_ones + n_ones,
            n_zeros=self.mu.n_zeros + n_zeros
        )

    def _proba(self, X):
        return np.prod(self.mu ** X * (1 - self.mu) ** (1 - X), axis=-1)

    def _draw(self, sample_size=1):
        if isinstance(self.mu, np.ndarray):
            return (
                self.mu > np.random.uniform(size=(sample_size, self.ndim))
            ).astype(np.int)
        elif isinstance(self.mu, Beta):
            return (
                self.mu.n_ones / (self.mu.n_ones + self.mu.n_zeros)
                > np.random.uniform(size=(sample_size, self.ndim))
            ).astype(np.int)
        elif isinstance(self.mu, Distribution):
            return (
                self.mu.draw(sample_size) > np.random.uniform(size=(sample_size, self.ndim))
            ).astype(np.int)
        else:
            raise AttributeError

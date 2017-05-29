import numpy as np
from prml.random.random import RandomVariable
from prml.random.beta import Beta


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu(prob)) = mu^x (1 - mu)^(1 - x)
    """

    def __init__(self, prob=None):
        """
        construct Bernoulli distribution

        Parameters
        ----------
        prob : (ndim,) np.ndarray or Beta
            probability of value 1
        """
        assert prob is None or isinstance(prob, (np.ndarray, Beta))
        self.prob = prob

    def __setattr__(self, name, value):
        if name is "prob":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert (value >= 0).all() and (value <= 1.).all()
                self.ndim = value.size
                object.__setattr__(self, name, value)
            elif isinstance(value, Beta):
                self.ndim = value.ndim
                self.prob_prior = value
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return "Bernoulli(prob={})".format(self.prob)

    @property
    def mean(self):
        return self.prob

    @property
    def var(self):
        return np.diag(self.prob * (1 - self.prob))

    def _ml(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones, (
            "{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        self.prob = np.mean(X, axis=0)

    def _map(self, X):
        assert isinstance(self.prob, Beta)
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        n_ones += self.prob.n_ones
        n_zeros += self.prob.n_zeros
        self.prob = (n_ones - 1) / (n_ones + n_zeros - 2)

    def _bayes(self, X):
        assert isinstance(self.prob, Beta)
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        self.prob = Beta(
            n_ones=self.prob.n_ones + n_ones,
            n_zeros=self.prob.n_zeros + n_zeros
        )

    def _pdf(self, X):
        return np.prod(self.prob ** X * (1 - self.prob) ** (1 - X), axis=-1)

    def _draw(self, sample_size=1):
        if isinstance(self.prob, np.ndarray):
            return (
                self.prob > np.random.uniform(size=(sample_size, self.ndim))
            ).astype(np.int)
        elif isinstance(self.prob, Beta):
            return (
                self.prob.n_ones / (self.prob.n_ones + self.prob.n_zeros)
                > np.random.uniform(size=(sample_size, self.ndim))
            ).astype(np.int)
        elif isinstance(self.prob, RandomVariable):
            return (
                self.prob.draw(sample_size)
                > np.random.uniform(size=(sample_size, self.ndim))
            ).astype(np.int)
        else:
            raise AttributeError

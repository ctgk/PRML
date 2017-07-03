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
        prob : np.ndarray or Beta
            probability of value 1 for each element
        """
        if isinstance(prob, Beta):
            self.prob_prior = prob
        self.prob = prob

    def __setattr__(self, name, value):
        if name is "prob":
            self.set_prob(value)
        else:
            object.__setattr__(self, name, value)

    def set_prob(self, prob):
        if isinstance(prob, (int, float, np.number)):
            if prob > 1 or prob < 0:
                raise ValueError(
                    "{prob} is out of the bound, prob must be in [0, 1]"
                )
            object.__setattr__(self, "prob", np.asarray(prob))
        elif isinstance(prob, np.ndarray):
            if (prob > 1).any() or (prob < 0).any():
                raise ValueError(
                    "{prob} is out of the bound, prob must be in [0, 1]"
                )
            object.__setattr__(self, "prob", prob)
        elif isinstance(prob, Beta):
            object.__setattr__(self, "prob", prob)
        else:
            if prob is not None:
                raise ValueError(
                    "{} is not acceptable for Bernoulli.prob".format(type(prob))
                )
            object.__setattr__(self, "prob", None)

    def __repr__(self):
        return "Bernoulli(prob={})".format(self.prob)

    @property
    def ndim(self):
        if hasattr(self.prob, "ndim"):
            return self.prob.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.prob, "size"):
            return self.prob.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.prob, "shape"):
            return self.prob.shape
        else:
            return None

    @property
    def mean(self):
        return self.prob

    @property
    def var(self):
        if self.prob is None:
            return None
        else:
            return self.prob * (1 - self.prob)

    def _ml(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones, (
            "{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        self.prob = np.mean(X, axis=0)

    def _map(self, X):
        assert isinstance(self.prob, Beta)
        assert X.shape[1:] == self.prob.shape
        n_ones = (X == 1).sum(axis=0)
        n_zeros = (X == 0).sum(axis=0)
        assert X.size == n_zeros.sum() + n_ones.sum(), (
            "{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        n_ones = n_ones + self.prob.n_ones
        n_zeros = n_zeros + self.prob.n_zeros
        self.prob = (n_ones - 1) / (n_ones + n_zeros - 2)

    def _bayes(self, X):
        assert isinstance(self.prob, Beta)
        assert X.shape[1:] == self.prob.shape
        n_ones = (X == 1).sum(axis=0)
        n_zeros = (X == 0).sum(axis=0)
        assert X.size == n_zeros.sum() + n_ones.sum(), (
            "input X must only has 0 or 1"
        )
        self.prob = Beta(
            n_ones=self.prob.n_ones + n_ones,
            n_zeros=self.prob.n_zeros + n_zeros
        )

    def _pdf(self, X):
        return np.prod(
            self.prob ** X * (1 - self.prob) ** (1 - X)
        )

    def _draw(self, sample_size=1):
        if isinstance(self.prob, np.ndarray):
            return (
                self.prob > np.random.uniform(size=(sample_size,) + self.shape)
            ).astype(np.int)
        elif isinstance(self.prob, Beta):
            return (
                self.prob.n_ones / (self.prob.n_ones + self.prob.n_zeros)
                > np.random.uniform(size=(sample_size,) + self.shape)
            ).astype(np.int)
        elif isinstance(self.prob, RandomVariable):
            return (
                self.prob.draw(sample_size)
                > np.random.uniform(size=(sample_size,) + self.shape)
            )

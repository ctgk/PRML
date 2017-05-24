import numpy as np
from prml.random.random import RandomVariable
from prml.random.dirichlet import Dirichlet


class Categorical(RandomVariable):
    """
    Categorical distribution
    p(x|mu(prob)) = prod_k mu_k^x_k
    """

    def __init__(self, prob=None):
        """
        construct categorical distribution

        Parameters
        ----------
        prob : (n_classes,) np.ndarray or Dirichlet
            probability of each class
        """
        self.prob = prob

    def __setattr__(self, name, value):
        if name is "prob":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert (value >= 0).all() and value.sum() == 1.
                self.n_classes = value.size
                object.__setattr__(self, name, value)
            elif isinstance(value, Dirichlet):
                self.n_classes = value.n_classes
                object.__setattr__(self, name, value)
            else:
                assert value is None, (
                    "prob must be either np.ndarray, Dirichlet, or None")
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return "Categorical(prob={})".format(self.prob)

    @property
    def mean(self):
        return self.prob

    def _check_input(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        assert (X.sum(axis=-1) == 1).all()

    def _ml(self, X):
        self._check_input(X)
        self.prob = np.mean(X, axis=0)

    def _map(self, X):
        self._check_input(X)
        assert isinstance(self.prob, Dirichlet)
        alpha = self.prob.alpha + X.sum(axis=0)
        self.prob = (alpha - 1) / (alpha - 1).sum()

    def _bayes(self, X):
        self._check_input(X)
        assert isinstance(self.prob, Dirichlet)
        self.prob = Dirichlet(
            concentration=self.prob.concentration + X.sum(axis=0)
        )

    def _call(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        assert np.allclose(X.sum(axis=-1), 1)
        return np.prod(self.prob ** X, axis=-1)

    def _draw(self, sample_size=1):
        return np.random.multinomial(1, self.prob, sample_size)

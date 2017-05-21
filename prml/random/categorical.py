import numpy as np
from prml.random.random import RandomVariable
from prml.random.dirichlet import Dirichlet


class Categorical(RandomVariable):
    """
    Categorical distribution
    p(x|mu) = \prod_k mu_k^x_k
    """

    def __init__(self, mu=None):
        """
        construct categorical distribution

        Parameters
        ----------
        mu : (n_classes,) np.ndarray or Dirichlet
            probability of each class
        """

        assert (mu is None or isinstance(mu, (np.ndarray, Dirichlet)))
        self.mu = mu

    def __setattr__(self, name, value):
        if name is "mu":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert (value >= 0).all() and value.sum() == 1.
                object.__setattr__(self, "n_classes", value.size)
                object.__setattr__(self, name, value)
            elif isinstance(value, Dirichlet):
                object.__setattr__(self, "n_classes", value.n_classes)
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return "Categorical(mu={})".format(self.mu)

    @property
    def mean(self):
        return self.mu

    def _check_input(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        assert (X.sum(axis=-1) == 1).all()

    def _ml(self, X):
        self._check_input(X)
        self.mu = np.mean(X, axis=0)

    def _map(self, X):
        self._check_input(X)
        assert isinstance(self.mu, Dirichlet)
        alpha = self.mu.alpha + X.sum(axis=0)
        self.mu = (alpha - 1) / (alpha - 1).sum()

    def _bayes(self, X):
        self._check_input(X)
        assert isinstance(self.mu, Dirichlet)
        self.mu = Dirichlet(
            alpha=self.mu.alpha + X.sum(axis=0)
        )

    def _proba(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones
        assert np.allclose(X.sum(axis=-1), 1)
        return np.prod(self.mu ** X, axis=-1)

    def _draw(self, sample_size=1):
        return np.random.multinomial(1, self.mu, sample_size)

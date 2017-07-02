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
        if isinstance(prob, Dirichlet):
            self.prob_prior = prob
        self.prob = prob

    def __setattr__(self, name, value):
        if name is "prob":
            self.set_prob(value)
        else:
            object.__setattr__(self, name, value)

    def set_prob(self, prob):
        if isinstance(prob, np.ndarray):
            if prob.ndim != 1:
                raise ValueError("prob must be 1 dimensional array")
            if (prob < 0).any():
                raise ValueError("prob must not have negative values")
            if prob.sum() != 1.:
                raise ValueError("sum of probs must equal to one")
            self.n_classes = prob.size
            object.__setattr__(self, "prob", prob)
        elif isinstance(prob, Dirichlet):
            self.n_classes = prob.size
            object.__setattr__(self, "prob", prob)
        else:
            if prob is not None:
                raise ValueError(
                    "{} is not acceptable for prob"
                    .format(type(prob))
                )
            object.__setattr__(self, "prob", None)

    def __repr__(self):
        return "Categorical(prob={})".format(self.prob)

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

    def _check_input(self, X):
        assert X.ndim == 2
        assert (X >= 0).all()
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

    def _pdf(self, X):
        self._check_input(X)
        return np.prod(self.prob ** X, axis=-1)

    def _draw(self, sample_size=1):
        return np.random.multinomial(1, self.prob, sample_size)

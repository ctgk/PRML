import numpy as np
from scipy.special import gamma
from prml.random.random import RandomVariable


class Dirichlet(RandomVariable):
    """
    Dirichlet distribution
    p(mu|alpha)
    = gamma(sum(alpha))
      * \prod_k mu_k ^ (alpha_k - 1)
      / gamma(alpha_1) / ... / gamma(alpha_K)
    """

    def __init__(self, alpha):
        """
        construct dirichlet distribution

        Parameters
        ----------
        alpha : (n_classes,) np.ndarray
            count of each class
        """
        assert isinstance(alpha, np.ndarray)
        assert alpha.ndim == 1
        assert (alpha >= 0).all()
        self.alpha = alpha
        self.n_classes = alpha.size

    def __repr__(self):
        return "Dirichlet(alpha={})".format(self.alpha)

    @property
    def mean(self):
        return self.alpha / self.alpha.sum()

    @property
    def var(self):
        a = self.alpha.sum()
        var = np.diag(self.alpha) * a - self.alpha[:, None] * self.alpha
        var = var / (a ** 2 * (a + 1))
        return var

    def _proba(self, mu):
        return (
            gamma(self.alpha.sum())
            * np.prod(mu ** (self.alpha - 1), axis=-1)
            / np.prod(gamma(self.alpha), axis=-1)
        )

    def _draw(self, sample_size=1):
        return np.random.dirichlet(self.alpha, sample_size)
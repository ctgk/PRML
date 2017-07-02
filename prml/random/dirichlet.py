import numpy as np
from scipy.special import gamma
from prml.random.random import RandomVariable


class Dirichlet(RandomVariable):
    """
    Dirichlet distribution
    p(mu|alpha(concentration))
    = gamma(sum(alpha))
      * prod_k mu_k ^ (alpha_k - 1)
      / gamma(alpha_1) / ... / gamma(alpha_K)
    """

    def __init__(self, concentration):
        """
        construct dirichlet distribution

        Parameters
        ----------
        concentration : (size,) np.ndarray
            count of each class
        """
        assert isinstance(concentration, np.ndarray)
        assert concentration.ndim == 1
        assert (concentration >= 0).all()
        self.concentration = concentration

    def __repr__(self):
        return "Dirichlet(concentration={})".format(self.concentration)

    @property
    def ndim(self):
        return self.concentration.ndim

    @property
    def size(self):
        return self.concentration.size

    @property
    def shape(self):
        return self.concentration.shape

    @property
    def mean(self):
        return self.concentration / self.concentration.sum()

    @property
    def var(self):
        a = self.concentration.sum()
        var = (
            np.diag(self.concentration) * a
            - self.concentration[:, None] * self.concentration)
        var = var / (a ** 2 * (a + 1))
        return var

    def _pdf(self, mu):
        return (
            gamma(self.concentration.sum())
            * np.prod(mu ** (self.concentration - 1), axis=-1)
            / np.prod(gamma(self.concentration), axis=-1)
        )

    def _draw(self, sample_size=1):
        return np.random.dirichlet(self.concentration, sample_size)

import numpy as np
from scipy.special import gamma


np.seterr(all="ignore")


class BetaDistribution(object):

    def __init__(self, pseudo_ones=1, pseudo_zeros=1):
        """
        construct beta distribution

        Parameters
        ----------
        pseudo_ones : float
            pseudo count of one
        pseudo_zeros : float
            pseudo count of zero
        """
        self.pseudo_ones = pseudo_ones
        self.pseudo_zeros = pseudo_zeros
        self.n_ones = pseudo_ones
        self.n_zeros = pseudo_zeros

    def fit(self, n_ones=None, n_zeros=None):
        """
        estimating parameter of posterior distribution of Beta(mu|a,b)

        Parameters
        ----------
        n_ones : float
            number of observed one
        n_zeros : float
            number of observed zero
        """
        self.n_ones = 0 if n_ones is None else n_ones
        self.n_zeros = 0 if n_zeros is None else n_zeros
        self.n_ones += self.pseudo_ones
        self.n_zeros += self.pseudo_zeros

    def fit_online(self, n_ones=None, n_zeros=None):
        """
        online estimation of posterior distribution Beta(mu|a,b)

        Parameters
        ----------
        n_ones : float
            number of observed one
        n_zeros : float
            number of observed zero
        """
        self.n_ones += 0 if n_ones is None else n_ones
        self.n_zeros += 0 if n_zeros is None else n_zeros

    def predict(self):
        """
        returns one or zero according to the posterior distribution

        Returns
        -------
        output : int
            prediction
        """
        return int(
            self.n_ones / (self.n_ones + self.n_zeros)
            > np.random.uniform())

    def probability(self, x):
        """
        probability denstiy function
        calculate posterior distribution beta(x|D)

        Parameters
        ----------
        x : ndarray
            input

        Returns
        -------
        output : float
            value of posterior distribution
        """
        return (
            gamma(self.n_ones + self.n_zeros)
            * np.power(x, self.n_ones - 1)
            * np.power(1 - x, self.n_zeros - 1)
            / gamma(self.n_ones)
            / gamma(self.n_zeros))

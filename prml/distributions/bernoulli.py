import numpy as np


class BernoulliDistribution(object):
    """
    Bernoulli distribution
    p(x|m) = m^x (1 - m)^(1 - x)
    """

    def __init__(self, mu=None):
        """
        construct bernoulli distribution

        Parameters
        ----------
        mu : (ndim,) np.ndarray
            mean of bernoulli distribution

        Attributes
        ----------
        ndim : int
            dimensionality
        """
        if mu is not None:
            self.ndim = mu.ndim
            self.mu = mu

    def fit(self, X):
        """
        maximum likelihood estimation of bernoulli distribution

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input data points

        Attributes
        ----------
        ndim : int
            dimensionality
        mu : (ndim,) np.ndarray
            mean of the distribution
        """
        if X.ndim == 1:
            X = X[:, None]
        self.ndim = np.size(X, 1)
        self.mu = np.mean(X, axis=0)

    def proba(self, X):
        """
        compute the probability distribution function
        Bern(x|m) = m^x * (1 - m)^(1 - x)

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input

        Returns
        -------
        p : (sample_size,) np.ndarray
            probability denstiy
        """
        if X.ndim == 1:
            X = X[:, None]
        return np.prod(self.mu ** X * (1 - self.mu) ** (1 - X), axis=-1)

    def draw(self, n=1):
        """
        draw sample from the distribution

        Parameters
        ----------
        n : int
            number of samples to draw from the distribution

        Returns
        -------
        sample : (n, ndim) np.ndarray
            generated sample
        """
        assert isinstance(n, int), type(n)
        return (
            self.mu > np.random.uniform(size=(n, self.ndim))).astype(np.int)

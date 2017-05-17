import numpy as np


class CategoricalDistribution(object):
    """
    Categorical distribution
    p(x|m) = \prod_k m_k^x_k
    """

    def __init__(self, mu=None):
        """
        construct categorical distribution

        Parameters
        ----------
        mu : (ndim,) np.ndarray
            mean parameter of the distribution

        Attributes
        ----------
        ndim : int
            dimensionality
        """
        if isinstance(mu, np.ndarray):
            assert np.sum(mu) == 1, np.sum(mu)
            self.mu = mu
            self.ndim = mu.ndim

    def fit(self, X):
        """
        maximum likelihood estimation of categorical distribution

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
        Cat(x|m) = \prod_k m_k^x_k

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input

        Returns
        -------
        p : (sample_size,) np.ndarray
            probability density
        """
        if X.ndim == 1:
            X = X[:, None]
        return np.prod(self.mu ** X, axis=-1)

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
        return np.random.choice(self.ndim, size=(n, self.ndim), p=self.mu)

import numpy as np


class GaussianDistribution(object):
    """
    the Gaussian distribution
    p(x|m,v) = exp(-0.5(x - m).T@v.inv@(x - m))/(sqrt(det(v))*(2pi)**(D/2))
    """

    def __init__(self, mean=None, var=None):
        """
        construct gaussian distribution

        Parameters
        ----------
        mean : (ndim,) ndarray
            mean of the Gaussian distribution
        var : (ndim, ndim) ndarray
            variance of the Gaussian distribution

        Attributes
        ----------
        ndim : int
            dimensionality
        """
        if mean is not None and var is not None:
            assert mean.ndim == 1, mean.ndim
            self.ndim = mean.shape[0]
            assert var.shape == (self.ndim, self.ndim), var.shape
            self.mean = mean
            self.var = var

    def fit(self, X):
        """
        maximum likelihood estimation of Gaussian distribution

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data points

        Attributes
        ----------
        ndim : int
            dimensionality
        mean : (n_features,) ndarray
            mean of gaussian distribution
        var : ndarray (n_features, n_features)
            variance of gaussian distribution
        """
        if X.ndim == 1:
            X = X[:, None]
        self.ndim = np.size(X, 1)
        self.mean = np.mean(X, axis=0)
        self.var = np.atleast_2d(np.cov(X, rowvar=False))

    def proba(self, X):
        """
        compute gauss function N(x|mu,Sigma)

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input

        Returns
        -------
        p : ndarray (sample_size,)
            probability density
        """
        if X.ndim == 1:
            X = X[:, None]
        d = X - self.mean
        precision = np.linalg.inv(self.var)
        return (
            np.exp(-0.5 * np.sum(d @ precision * d, axis=-1))
            * np.sqrt(np.linalg.det(precision))
            / np.power(2 * np.pi, 0.5 * self.ndim))

    def draw(self, n=1):
        """
        draw sample from this distribution

        Parameters
        ----------
        n : int
            number of samples to draw from the distribution

        Returns
        -------
        sample : (n, ndim) ndarray
            generated sample
        """
        assert isinstance(n, int), type(n)
        return np.random.multivariate_normal(self.mean, self.var, n)

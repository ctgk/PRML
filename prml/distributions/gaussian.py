import numpy as np


class GaussianDistribution(object):

    def fit(self, X):
        """
        maximum likelihood estimation of Gaussian distribution

        Parameters
        ----------
        X : (sample_size, n_features)
            input data points

        Attributes
        ----------
        mean : ndarray (n_features,)
            mean of gaussian distribution
        var : ndarray (n_features, n_features)
            variance of gaussian distribution
        """
        if X.ndim == 1:
            X = X[:, None]
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
            / np.power(2 * np.pi, 0.5 * np.size(X, 1)))

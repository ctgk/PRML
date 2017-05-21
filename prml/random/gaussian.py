import numpy as np
from prml.random.random import RandomVariable
from prml.random.gamma import Gamma


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu, var)
    = exp{-0.5 * (x - mu)^T @ var^-1 @ (x - mu)}
      / (2pi)^(D/2) / |var|^0.5
    """

    def __init__(self, mu=None, var=None, precision=None):
        assert mu is None or isinstance(mu, (int, float, np.ndarray, Gaussian))
        assert var is None or isinstance(var, (int, float, np.ndarray))
        assert precision is None or isinstance(precision, (int, float, np.ndarray, Gamma))
        self.mu = mu
        if var is not None:
            self.var = var
        elif precision is not None:
            self.precision = precision
        else:
            self.var = None
            self.precision = None

    def __setattr__(self, name, value):
        if name is "mu":
            if isinstance(value, (int, float)):
                object.__setattr__(self, "ndim", 1)
                object.__setattr__(self, name, np.array([value]))
            elif isinstance(value, np.ndarray):
                assert value.ndim == 1
                object.__setattr__(self, "ndim", value.size)
                object.__setattr__(self, name, value)
            elif isinstance(value, Gaussian):
                object.__setattr__(self, "ndim", value.ndim)
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        elif name is "var":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, value * np.eye(self.ndim))
                object.__setattr__(self, "precision", 1 / self.var)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "precision", np.linalg.inv(value))
            else:
                object.__setattr__(self, name, None)
        elif name is "precision":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, value * np.eye(self.ndim))
                object.__setattr__(self, "var", 1 / self.precision)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "var", np.linalg.inv(value))
            elif isinstance(value, Gamma):
                assert self.ndim == 1
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        if hasattr(self, "var"):
            return "Gaussian(\nmu={0},\nvar=\n{1}\n)".format(self.mean, self.var)
        else:
            return "Gaussian(\nmu={0},\nprecision=\n{1}\n)".format(self.mean, self.precision)

    @property
    def mean(self):
        return self.mu

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.var = np.atleast_2d(np.cov(X, rowvar=False, bias=True))

    def _map(self, X):
        assert isinstance(self.mu, Gaussian)
        assert isinstance(self.var, np.ndarray)
        N = len(X)
        self.mu = np.linalg.solve(
            self.mu.precision + N * self.precision,
            self.mu.precision @ self.mu.mu + self.precision @ np.mean(X, axis=0) * N)

    def _bayes(self, X):
        N = len(X)
        if isinstance(self.mu, Gaussian) and isinstance(self.precision, np.ndarray):
            mu = np.mean(X, axis=0)
            S = np.linalg.inv(self.mu.precision + N * self.precision)
            self.mu = Gaussian(
                mu=S @ (self.mu.precision @ self.mu.mu + self.precision @ mu * N),
                var=S
            )
        elif isinstance(self.mu, np.ndarray) and isinstance(self.precision, Gamma):
            assert np.size(X, 1) == 1
            self.precision = Gamma(
                a=self.precision.a + 0.5 * N,
                b=self.precision.b + 0.5 * N * np.var(X)
            )
        elif isinstance(self.mu, Gaussian) and isinstance(self.precision, Gamma):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _proba(self, X):
        d = X - self.mean
        return (
            np.exp(-0.5 * np.sum(d @ self.precision * d, axis=-1))
            * np.sqrt(np.linalg.det(self.precision))
            / np.power(2 * np.pi, 0.5 * self.ndim))

    def _draw(self, sample_size=1):
        return np.random.multivariate_normal(self.mean, self.var, sample_size)


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
        if mean is not None:
            assert mean.ndim == 1, mean.ndim
            self.ndim = mean.size
            self.mean = mean
        if var is not None:
            assert var.shape == (self.ndim, self.ndim), var.shape
            self.var = var

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

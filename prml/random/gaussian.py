import numpy as np
from prml.random.random import RandomVariable
from prml.random.gamma import Gamma


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu(mean), var)
    = exp{-0.5 * (x - mu)^T @ var^-1 @ (x - mu)}
      / (2pi)^(D/2) / |var|^0.5
    """

    def __init__(self, mean=None, var=None, precision=None):
        self.mean = mean
        if var is not None:
            self._var = var
        elif precision is not None:
            self._precision = precision
        else:
            self._var = None
            self._precision = None

    def __setattr__(self, name, value):
        if name is "mean":
            if isinstance(value, (int, float)):
                self.ndim = 1
                object.__setattr__(self, name, np.array([value]))
            elif isinstance(value, np.ndarray):
                assert value.ndim == 1
                self.ndim = value.size
                object.__setattr__(self, name, value)
            elif isinstance(value, Gaussian):
                self.ndim = value.ndim
                self.mean_prior = value
                object.__setattr__(self, name, value)
            else:
                assert value is None, (
                    "mean must be either"
                    "int, float, np.ndarray, Gaussian, or None")
                object.__setattr__(self, name, None)
        elif name is "_var":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, value)
                object.__setattr__(self, "_precision", 1 / self.var)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "_precision", np.linalg.inv(value))
            else:
                assert value is None, (
                    "var must be either int, float, np.ndarray, or None"
                )
                object.__setattr__(self, name, None)
        elif name is "_precision":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, value)
                object.__setattr__(self, "_var", 1 / self.precision)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "_var", np.linalg.inv(value))
            elif isinstance(value, Gamma):
                assert self.ndim == 1
                self.precision_prior = value
                object.__setattr__(self, name, value)
            else:
                assert value is None, (
                    "precision must be either"
                    "int, float, np.ndarray, Gamma, or None"
                )
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        if hasattr(self, "var"):
            return "Gaussian(\nmean={0.mean},\nvar=\n{0.var}\n)".format(self)
        else:
            return (
                "Gaussian(\nmean={0.mean},\nprecision=\n{0.precision}\n)"
                .format(self)
            )

    @property
    def var(self):
        if isinstance(self._var, (int, float)):
            return self._var * np.eye(self.ndim)
        else:
            return self._var

    @property
    def precision(self):
        if isinstance(self._precision, (int, float)):
            return self._precision * np.eye(self.ndim)
        else:
            return self._precision

    def _ml(self, X):
        self.mean = np.mean(X, axis=0)
        self._var = np.atleast_2d(np.cov(X.T, bias=True))

    def _map(self, X):
        assert isinstance(self.mean, Gaussian)
        assert isinstance(self.var, np.ndarray)
        N = len(X)
        mu = np.mean(X, 0)
        self.mean = np.linalg.solve(
            self.mean.precision + N * self.precision,
            self.mean.precision @ self.mean.mean + self.precision @ mu * N)

    def _bayes(self, X):
        N = len(X)
        mean_is_gaussian = isinstance(self.mean, Gaussian)
        mean_is_ndarray = isinstance(self.mean, np.ndarray)
        precison_is_ndarray = isinstance(self.precision, np.ndarray)
        precision_is_gamma = isinstance(self.precision, Gamma)
        if mean_is_gaussian and precison_is_ndarray:
            mu = np.mean(X, axis=0)
            S = np.linalg.inv(self.mean.precision + N * self.precision)
            self.mean = Gaussian(
                mean=S @ (
                    self.mean.precision @ self.mean.mean
                    + self.precision @ mu * N),
                var=S
            )
        elif mean_is_ndarray and precision_is_gamma:
            assert np.size(X, 1) == 1
            self._precision = Gamma(
                shape=self.precision.shape + 0.5 * N,
                rate=self.precision.rate + 0.5 * N * np.var(X)
            )
        elif mean_is_gaussian and precision_is_gamma:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _call(self, X):
        d = X - self.mean
        return (
            np.exp(-0.5 * np.sum(d @ self.precision * d, axis=-1))
            * np.sqrt(np.linalg.det(self.precision))
            / np.power(2 * np.pi, 0.5 * self.ndim))

    def _draw(self, sample_size=1):
        return np.random.multivariate_normal(self.mean, self.var, sample_size)

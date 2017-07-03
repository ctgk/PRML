import numpy as np
from prml.random.random import RandomVariable
from prml.random.gamma import Gamma


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu(mean), var)
    = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self, mean=None, var=None, precision=None):
        self.mean = mean
        if isinstance(mean, RandomVariable):
            self.mean_prior = mean
        if var is not None:
            self.var_ = var
        elif precision is not None:
            self.precision_ = precision
            if isinstance(precision, RandomVariable):
                self.precision_prior = precision
        else:
            self.var_ = None
            self.precision_ = None

    def __setattr__(self, name, value):
        if name is "mean":
            self.set_mean(value)
        elif name is "var_":
            self.set_var_(value)
        elif name is "precision_":
            self.set_precision_(value)
        else:
            object.__setattr__(self, name, value)

    def set_mean(self, mean):
        if isinstance(mean, (int, float, np.number)):
            object.__setattr__(self, "mean", np.array(mean))
        elif isinstance(mean, np.ndarray):
            object.__setattr__(self, "mean", mean)
        elif isinstance(mean, Gaussian):
            object.__setattr__(self, "mean", mean)
        else:
            assert mean is None, (
                "mean must be either"
                "int, float, np.ndarray, Gaussian, or None"
            )
            object.__setattr__(self, "mean", None)

    def set_var_(self, var_):
        if isinstance(var_, (int, float, np.number)):
            object.__setattr__(self, "var_", var_)
            object.__setattr__(self, "precision_", 1 / var_)
        elif isinstance(var_, np.ndarray):
            assert var_.shape == self.shape
            object.__setattr__(self, "var_", var_)
            object.__setattr__(self, "precision_", 1 / var_)
        else:
            assert var_ is None, (
                "var must be either int, float, np.ndarray, or None"
            )
            object.__setattr__(self, "var_", None)
            object.__setattr__(self, "precision_", None)

    def set_precision_(self, precision_):
        if isinstance(precision_, (int, float, np.number)):
            object.__setattr__(self, "var_", 1 / precision_)
            object.__setattr__(self, "precision_", precision_)
        elif isinstance(precision_, np.ndarray):
            assert precision_.shape == self.shape
            object.__setattr__(self, "var_", 1 / precision_)
            object.__setattr__(self, "precision_", precision_)
        elif isinstance(precision_, Gamma):
            object.__setattr__(self, "precision_", precision_)
        else:
            assert precision_ is None, (
                "precision must be either "
                "int, float, np.ndarray, Gamma, or None"
            )
            object.__setattr__(self, "precision_", None)

    def __repr__(self):
        reprval = "Gaussian("
        if hasattr(self, "var_"):
            name = "var"
            value = "{}".format(self.var_)
        else:
            name = "precision"
            value = "{}".format(self.precision_)
        if self.ndim is None or self.ndim == 0:
            reprval += "mean={0.mean}, {1}={2})".format(self, name, value)
        elif self.ndim == 1:
            reprval += "\nmean={0.mean},\n{1}={2})".format(self, name, value)
        else:
            reprval += "\nmean=\n{0.mean},\n{1}=\n{2}\n)".format(self, name, value)
        return reprval

    @property
    def ndim(self):
        if hasattr(self.mean, "ndim"):
            return self.mean.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mean, "size"):
            return self.mean.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mean, "shape"):
            return self.mean.shape
        else:
            return None

    @property
    def var(self):
        if isinstance(self.var_, (int, float, np.number)):
            if self.shape == ():
                return np.array(self.var_)
            else:
                return self.var_ * np.ones(self.shape)
        else:
            return self.var_

    @property
    def precision(self):
        if isinstance(self.precision_, (int, float, np.number)):
            if self.shape == ():
                return np.array(self.precision_)
            else:
                return self.precision_ * np.ones(self.shape)
        else:
            return self.precision_

    def _ml(self, X):
        self.mean = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)

    def _map(self, X):
        assert isinstance(self.mean, Gaussian)
        assert isinstance(self.var, np.ndarray)
        N = len(X)
        mu = np.mean(X, 0)
        self.mean = (
            (self.precision * self.mean.mean + N * self.mean.precision * mu)
            / (N * self.mean.precision + self.precision)
        )

    def _bayes(self, X):
        N = len(X)
        mean_is_gaussian = isinstance(self.mean, Gaussian)
        mean_is_ndarray = isinstance(self.mean, np.ndarray)
        precision_is_ndarray = isinstance(self.precision, np.ndarray)
        precision_is_gamma = isinstance(self.precision, Gamma)
        if mean_is_gaussian and precision_is_ndarray:
            mu = np.mean(X, 0)
            precision = self.mean.precision + N * self.precision
            self.mean = Gaussian(
                mean=(self.mean.mean * self.mean.precision + N * mu * self.precision) / precision,
                precision=precision
            )
        elif mean_is_ndarray and precision_is_gamma:
            var = np.var(X, axis=0)
            shape = self.precision.shape_ + 0.5 * N
            rate = self.precision.rate + 0.5 * N * var
            self.precision_ = Gamma(shape, rate)
        elif mean_is_gaussian and precision_is_gamma:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _pdf(self, X):
        d = X - self.mean
        return (
            np.exp(-0.5 * self.precision * d ** 2) / np.sqrt(2 * np.pi * self.var)
        )

    def _draw(self, sample_size=1):
        return np.random.normal(
            loc=self.mean,
            scale=np.sqrt(self.var),
            size=(sample_size,) + self.shape
        )

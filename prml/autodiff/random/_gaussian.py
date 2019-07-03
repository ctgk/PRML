from itertools import zip_longest

import numpy as np

from prml.autodiff._core._config import config
from prml.autodiff._core._function import _Function, broadcast_to
from prml.autodiff.linalg._cholesky import cholesky


class _Gaussian(_Function):
    enable_auto_broadcast = True

    def _forward(self, mean, std):
        self.eps = np.random.randn(*mean.shape).astype(config.dtype)
        return mean + std * self.eps

    def _backward(self, delta, mean, std):
        return delta, delta * self.eps


def gaussian(mean, std, size=None):
    """
    random sample from gaussian distribution

    Parameters
    ----------
    mean : array_like
        mean of the distribution to sample from
    std : array_like
        standard deviation of the distribution to sample from
    size : tuple, optional
        size of sample, by default None

    Returns
    -------
    Array
        sample from the distribution
    """
    if size is not None:
        mean = broadcast_to(mean, size)
        std = broadcast_to(std, size)
    return _Gaussian().forward(mean, std)


class _GaussianLogPDF(_Function):
    enable_auto_broadcast = True
    log2pi = np.log(2 * np.pi)

    def _forward(self, x, mean, std):
        self.mahalanobis_squared = np.square((x - mean) / std)
        log_pdf = -0.5 * (
            self.mahalanobis_squared
            + 2 * np.log(std)
            + self.log2pi
        )
        return log_pdf

    def _backward(self, delta, x, mean, std):
        dmean = delta * (x - mean) / np.square(std)
        dx = -dmean
        dstd = delta * (self.mahalanobis_squared - 1) / std
        return dx, dmean, dstd


def gaussian_logpdf(x, mean, std):
    return _GaussianLogPDF().forward(x, mean, std)


class _MultivariateGaussian(_Function):

    def _forward(self, mean, cholesky_cov):
        self.eps = np.random.standard_normal(size=mean.shape).astype(config.dtype)
        return mean + np.einsum("...ij,...j->...i", cholesky_cov, self.eps)

    def _backward(self, delta, mean, cholesky_cov):
        dmean = delta
        dcholesky_cov = np.einsum("...i,...j->...ij", delta, self.eps)
        return dmean, dcholesky_cov


def multivariate_gaussian(mean, covariance, size=None):
    """
    random sample from multivariate gaussian distribution

    Parameters
    ----------
    mean : array_like (..., D)
        mean of the distribution to sample from
    covariance : array_like (..., D, D)
        covariance of the distribution to sample from
    size : tuple (..., ..., D), optional
        size of sample, by default None

    Returns
    -------
    Array
        sample from the distribution
    """
    cholesky_cov = cholesky(covariance)
    if size is None:
        size_reversed = tuple(
            max(x, y) for x, y
            in zip_longest(
                reversed(mean.shape),
                reversed(covariance.shape[:-1]),
                1
            )
        )
        size = tuple(reversed(size_reversed))
    mean = broadcast_to(mean, size)
    cholesky_cov = broadcast_to(cholesky_cov, size + (size[-1],))
    return _MultivariateGaussian().forward(mean, cholesky_cov)

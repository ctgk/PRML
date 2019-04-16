import numpy as np
import scipy.special as sp
from prml.nn.array.array import asarray
from prml.nn.distribution.distribution import Distribution
from prml.nn.function import Function
from prml.nn.math.log import log
from prml.nn.math.sqrt import sqrt
from prml.nn.math.square import square
from prml.nn.random.normal import normal


class Gaussian(Distribution):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = asarray(mean)
        self.std = asarray(std)
        assert((self.std.value > 0).all())

    def forward(self):
        if np.prod(self.mean.shape) > np.prod(self.std.shape):
            eps = normal(0, 1, self.mean.shape)
        else:
            eps = normal(0, 1, self.std.shape)
        return self.mean + self.std * eps

    def _log_pdf(self, x):
        return GaussianLogPDF().forward(x, self.mean, self.std)


class GaussianLogPDF(Function):
    enable_auto_broadcast = True
    log2pi = np.log(2 * np.pi)

    def _forward(self, x, mean, std):
        self.mahalanobis_distance = np.square((x - mean) / std)
        log_pdf = -0.5 * (
            self.mahalanobis_distance
            + 2 * np.log(std)
            + self.log2pi
        )
        return log_pdf

    def _backward(self, delta, x, mean, std):
        dx = -0.5 * delta * (x - mean) / np.square(std)
        dmean = -dx
        dstd = (self.mahalanobis_distance - 1) / std
        return dx, dmean, dstd


class GaussianRadial(Distribution):

    def __init__(self, std, ndim: int):
        super().__init__()
        self.std = asarray(std)
        self.ndim = ndim
        assert((self.std.value >= 0).all())

    def forward(self):
        eps = normal(0, 1, (self.ndim,) + self.std.shape)
        return sqrt(square(self.std * eps).sum(axis=0))

    def _log_pdf(self, x):
        return (
            (self.ndim - 1) * log(x)
            - 0.5 * square(x / self.std)
            - self.ndim * log(self.std)
            - np.log(sp.gamma(0.5 * self.ndim))
        )

import numpy as np
from scipy.special import digamma, loggamma

from prml.autodiff._core._function import broadcast_to, _Function
from prml.autodiff.random._gamma import gamma


def beta(a, b, size=None):
    if size is not None:
        a = broadcast_to(a, size)
        b = broadcast_to(b, size)
    x = gamma(a, 1)
    y = gamma(b, 1)
    return x / (x + y)


class _BetaLogPDF(_Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, a, b):
        return (
            loggamma(a + b)
            - loggamma(a)
            - loggamma(b)
            + (a - 1) * np.log(x)
            + (b - 1) * np.log(1 - x)
        )

    @staticmethod
    def _backward(delta, x, a, b):
        dx = (a - 1) / x - (b - 1) / (1 - x)
        da = digamma(a + b) - digamma(a) + np.log(x)
        db = digamma(a + b) - digamma(b) + np.log(1 - x)
        return dx, da, db


def beta_logpdf(x, a, b):
    return _BetaLogPDF().forward(x, a, b)

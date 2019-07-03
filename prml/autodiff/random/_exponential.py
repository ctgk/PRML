import numpy as np

from prml.autodiff._core._function import _Function
from prml.autodiff._core._function import broadcast_to


class _Exponential(_Function):

    def _forward(self, rate: np.ndarray):
        eps = np.random.standard_exponential(rate.shape)
        self.output = eps / rate
        return self.output

    def _backward(self, delta, rate):
        drate = -delta * self.output / rate
        return drate


def exponential(rate, size=None):
    r"""
    Exponential distribution

    .. math::

        p(x|\lambda) = \lambda e^{-\lambda x}

    Parameters
    ----------
    rate : array_like
        rate parameter
    size : tuple, optional
        size of sample, by default None

    Returns
    -------
    Array
        sample of exponential distriubtion
    """
    if size is not None:
        rate = broadcast_to(rate, size)
    return _Exponential().forward(rate)


class _ExponentialLogPDF(_Function):

    @staticmethod
    def _forward(x, rate):
        return -rate * x + np.log(rate)

    def _backward(delta, x, rate):
        dx = -delta * rate
        drate = delta * (1 / rate - x)
        return dx, drate


def exponential_logpdf(x, rate):
    return _ExponentialLogPDF().forward(x, rate)

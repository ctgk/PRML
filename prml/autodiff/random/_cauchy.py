import numpy as np

from prml.autodiff._core._function import broadcast_to, _Function


class _Cauchy(_Function):
    enable_auto_broadcast = True

    def _forward(self, loc, scale):
        self.eps = np.random.standard_cauchy(size=loc.shape)
        self.output = loc + self.eps * scale
        return self.output

    def _backward(self, delta, loc, scale):
        dloc = delta
        dscale = delta * self.eps
        return dloc, dscale


def cauchy(loc, scale, size=None):
    r"""
    Cauchy distribution aka Lorentz distribution

    .. math::

        p(x|\mu,\gamma) = {1\over \pi\gamma*(1+({x-\mu\over\gamma})^2)}

    Parameters
    ----------
    loc : array_like
        location parameter :math:`\mu`
    scale : array_like
        scale parameter :math:`\gamma`
    size : tuple, optional
        sample size, by default None

    Returns
    -------
    Array
        sample from Cauchy distribution
    """
    if size is not None:
        loc = broadcast_to(loc, size)
        scale = broadcast_to(scale, size)
    return _Cauchy().forward(loc, scale)


class _CauchyLogPDF(_Function):

    @staticmethod
    def _forward(x, loc, scale):
        return -np.log(np.pi * scale * (1 + np.square((x - loc) / scale)))

    @staticmethod
    def _backward(delta, x, loc, scale):
        d = x - loc
        d2 = d ** 2
        scale2 = scale ** 2
        dloc = 2 * d / (d2 + scale2)
        dscale = (d2 - scale2) / (scale2 + d2) / scale
        return -dloc, dloc, dscale


def cauchy_logpdf(x, loc, scale):
    return _CauchyLogPDF().forward(x, loc, scale)

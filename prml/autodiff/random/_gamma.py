import numpy as np
import scipy.special as sp

from prml.autodiff._core._function import broadcast_to, _Function


class _Gamma(_Function):
    enable_auto_broadcast = True

    def _forward(self, shape, rate):
        self.output = np.random.gamma(shape, 1 / rate)
        return self.output

    def _backward(self, delta, shape, rate):
        psishape = sp.digamma(shape)
        psi1shape = sp.polygamma(1, shape)
        sqrtpsi1shape = np.sqrt(psi1shape)
        psi2shape = sp.polygamma(2, shape)
        eps = (np.log(self.output) - psishape + np.log(rate)) / sqrtpsi1shape
        dshape = delta * self.output * (
            0.5 * eps * psi2shape / sqrtpsi1shape + psi1shape)
        drate = -delta * self.output / rate
        return dshape, drate


def gamma(shape, rate, size=None):
    if size is not None:
        shape = broadcast_to(shape, size)
        rate = broadcast_to(rate, size)
    return _Gamma().forward(shape, rate)

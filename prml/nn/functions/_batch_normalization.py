import numpy as np

from prml.autodiff._core._function import _Function


class _BatchNormalization(_Function):

    def __init__(self, eps=1e-7):
        self.eps = eps

    def _forward(self, x):
        self.mean = x.mean(axis=0)
        self.xc = x - self.mean
        self.var = np.mean(self.xc ** 2, axis=0)
        self.std = np.sqrt(self.var + self.eps)
        return self.xc / self.std

    def _backward(self, delta, x):
        dxc = (
            delta / self.std
            - self.xc * np.mean((delta * self.xc) / (self.std ** 3), axis=0)
        )
        return dxc - np.mean(dxc, axis=0)

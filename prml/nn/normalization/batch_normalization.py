import numpy as np
from prml.nn.array.ones import ones
from prml.nn.array.zeros import zeros
from prml.nn.config import config
from prml.nn.function import Function
from prml.nn.network import Network


class BatchNormalizationFunction(Function):

    def _forward(self, x):
        self.mean = x.mean(axis=0)
        self.xc = x - self.mean
        self.var = np.mean(self.xc ** 2, axis=0)
        self.std = np.sqrt(self.var + 1e-7)
        return self.xc / self.std

    def _backward(self, delta, x):
        # dstd = -np.mean((delta * self.xc) / (self.std ** 2), axis=0)
        dxc = delta / self.std - self.xc * np.mean((delta * self.xc) / (self.std ** 3), axis=0)
        return dxc - np.mean(dxc, axis=0)

        # dstd = -np.mean((delta * self.xc) / (self.std ** 2), axis=0)
        # dxc = delta / self.std + self.xc * dstd / self.std
        # return dxc - np.mean(dxc, axis=0)

        # dxn = delta
        # dxc = dxn / self.std
        # dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)
        # dvar = 0.5 * dstd / self.std
        # dxc += 2.0 * self.xc * dvar / delta.shape[0]
        # dmu = np.sum(dxc, axis=0)
        # dx = dxc - dmu / delta.shape[0]
        # return dx


class BatchNormalization(Network):

    def __init__(self, ndim, scale=None, bias=None, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        with self.set_parameter():
            self.mean = zeros(ndim)
            self.var = ones(ndim)

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        if config.is_updating_bn:
            func = BatchNormalizationFunction()
            out = func.forward(x)
            self.mean.value = self.momentum * self.mean.value + (1 - self.momentum) * func.mean
            self.var.value = self.momentum * self.var.value + (1 - self.momentum) * func.var
            del func.mean
            del func.var
        else:
            xc = x - self.mean
            out = xc / np.sqrt(self.var.value + 1e-7)
        return out.reshape(*shape)

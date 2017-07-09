import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function
from prml.function.array.broadcast import broadcast_to


class Gaussian(Function):
    """
    gaussian sampling
    """

    def _check_input(self, mean, std):
        mean = self._convert2tensor(mean)
        std = self._convert2tensor(std)
        if mean.shape != std.shape:
            shape = np.broadcast(mean.value, std.value).shape
            if mean.shape != shape:
                mean = broadcast_to(mean, shape)
            if std.shape != shape:
                std = broadcast_to(std, shape)
        return mean, std

    def _forward(self, mean, std):
        mean, std = self._check_input(mean, std)
        self.mean = mean
        self.std = std
        self.eps = np.random.normal(size=mean.shape)
        output = mean.value + std.value * self.eps
        return Tensor(output, function=self)

    def _backward(self, delta):
        dmean = delta
        dstd = delta * self.eps
        self.mean.backward(dmean)
        self.std.backward(dstd)


def gaussian(mean, std):
    """
    gaussian sampling function
    """
    return Gaussian().forward(mean, std)

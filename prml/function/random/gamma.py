import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function
from prml.function.array.broadcast import broadcast_to


class Gamma(Function):
    """
    sampling from Gamma distribution
    """

    def _check_input(self, shape_, rate):
        shape_ = self._convert2tensor(shape_)
        rate = self._convert2tensor(rate)
        if shape_.shape != rate.shape:
            shape = np.broadcast(shape_.value, rate.value).shape
            if shape_.shape != shape:
                shape_ = broadcast_to(shape_, shape)
            if rate.shape != shape:
                rate = broadcast_to(rate, shape)
        return shape_, rate

    def _forward(self, shape_, rate):
        shape_, rate = self._check_input(shape_, rate)
        self.shape_ = shape_
        self.rate = rate
        raise NotImplementedError


def gamma(shape, rate):
    """
    sampling from Gamma distribution
    p(x|a(shape), b(rate))
    = b^a x^(a-1) exp(-bx) / gamma(a)
    """
    return Gamma().forward(shape, rate)

import numpy as np

from prml import autodiff
from prml.nn.functions._batch_normalization import _BatchNormalization
from prml.nn.layers._layer import _Layer
from prml.nn._nnconfig import nnconfig


class BatchNormalization(_Layer):

    def __init__(
        self,
        ndim_in: int,
        ndim_out: int,
        momentum: float = 0.9,
        has_bias: bool = True,
        has_scale: bool = True,
        eps: float = 1e-7
    ):
        super().__init__()
        with self.initialize():
            self.mean = autodiff.zeros(ndim_in)
            self.mean.requires_grad = False
            self.var = autodiff.ones(ndim_in)
            self.var.requires_grad = False
        if has_bias:
            self.initialize_bias(ndim_out)
        if has_scale:
            with self.initialize():
                self.scale = autodiff.ones(ndim_out)
        else:
            self.scale = None
        self.func = _BatchNormalization(eps)
        self.eps = eps

    def interpolate(self, value_new, value_prev):
        return self.momentum * value_prev + (1 - self.momentum) * value_new

    def _forward(self, x):
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        if nnconfig.is_updating_bn:
            out = self.func.forward(x)
            self.mean.value = self.interpolate(self.func.mean, self.mean.value)
            self.var.value = self.interpolate(self.func.var, self.var.value)
        else:
            xc = x - self.mean
            out = xc / np.sqrt(self.var.value + self.eps)
        out = out.reshape(*shape)
        if self.scale is not None:
            out = self.scale * out
        return out

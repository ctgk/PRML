import numpy as np

from prml.autodiff._core._array import Array, asarray
from prml.autodiff._core._config import config


class _Function(object):
    enable_auto_broadcast = False

    def forward(self, *args, **kwargs):
        self.args = [self._convert2array(arg) for arg in args]
        if self.enable_auto_broadcast:
            self.args = self._autobroadcast(*self.args)
        self.kwargs = kwargs
        out = self._forward(*tuple(arg.value for arg in self.args), **kwargs)
        out = Array(out)
        if config.enable_backprop:
            out.add_parent(self)
        return out

    def backward(self, delta, backprop_taskmanager):
        dargs = self._backward(
            delta,
            *tuple(arg.value for arg in self.args),
            **self.kwargs
        )
        if isinstance(dargs, tuple):
            for arg, darg in zip(self.args, dargs):
                backprop_taskmanager.add_task(arg)
                arg._accumulate_gradient_from_child(darg)
        else:
            backprop_taskmanager.add_task(self.args[0])
            self.args[0]._accumulate_gradient_from_child(dargs)

    def _out_depth(self):
        return max([arg._depth for arg in self.args]) + 1

    @staticmethod
    def _autobroadcast(*args):
        return broadcast(*args)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _backward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _convert2array(arg):
        if not isinstance(arg, Array):
            return asarray(arg)
        else:
            return arg


class _BroadcastTo(_Function):

    def __init__(self, shape):
        self.shape = shape

    def _forward(self, x):
        output = np.broadcast_to(x, self.shape)
        return output

    @staticmethod
    def _backward(delta, x):
        dx = delta
        xdim = getattr(x, "ndim", 0)
        xshape = getattr(x, "shape", ())
        if delta.ndim != xdim:
            dx = dx.sum(axis=tuple(range(dx.ndim - xdim)))
            if isinstance(dx, np.number):
                dx = np.array(dx)
        axis = tuple(i for i, len_ in enumerate(xshape) if len_ == 1)
        if axis:
            dx = dx.sum(axis=axis, keepdims=True)
        return dx


def broadcast_to(x, shape):
    """
    Broadcast an array to an new shape

    Parameters
    ----------
    x : array_like
        input array to broadcast its shape
    shape : tuple
        target shape

    Returns
    -------
    Array
        input array broadcasted to target shape
    """
    if getattr(x, "shape", ()) != shape:
        return _BroadcastTo(shape).forward(x)
    return x


def broadcast(*args):
    """
    broadcast list of arrays to make them have the same shape

    Parameters
    ----------
    args : array_like
        arrays to be aligned

    Returns
    -------
    tuple
        tuple of arrays whose shapes are aligned
    """
    shape = np.broadcast(*(arg.value for arg in args)).shape
    args = tuple(
        arg if arg.shape == shape else _BroadcastTo(shape).forward(arg)
        for arg in args)
    return args

import numpy as np

from prml.autodiff._core._config import config


class Array(object):
    __array_ufunc__ = None

    def __init__(self, value):
        self.value = np.atleast_1d(value)
        self._parent = None
        self.grad = None
        self._gradtmp = None
        self._depth = 0

    def add_parent(self, parent):
        self._parent = parent
        self._depth = parent._out_depth()

    def __repr__(self):
        return f"Array(shape={self.value.shape}, dtype={self.value.dtype})"

    def __len__(self):
        return len(self.value)

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def dtype(self):
        return self.value.dtype

    def cleargrad(self):
        self.grad = None
        self._gradtmp = None

    def backprop(self, grad=None):
        raise NotImplementedError

    def update_grad(self, grad):
        if self.grad is None:
            self.grad = np.copy(grad)
        else:
            self.grad += grad

    def _accumulate_gradient_from_child(self, grad):
        if grad is None:
            return
        assert(grad.shape == self.shape)
        if self._gradtmp is None:
            self._gradtmp = np.copy(grad)
        else:
            self._gradtmp += grad

    def __add__(self, arg):
        raise NotImplementedError

    def __radd__(self, arg):
        raise NotImplementedError

    def __truediv__(self, arg):
        raise NotImplementedError

    def __rtruediv__(self, arg):
        raise NotImplementedError

    def __matmul__(self, arg):
        raise NotImplementedError

    def __rmatmul__(self, arg):
        raise NotImplementedError

    def __mul__(self, arg):
        raise NotImplementedError

    def __rmul__(self, arg):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pow__(self, arg):
        raise NotImplementedError

    def __rpow__(self, arg):
        raise NotImplementedError

    def __sub__(self, arg):
        raise NotImplementedError

    def __rsub__(self, arg):
        raise NotImplementedError

    def flatten(self):
        raise NotImplementedError

    def inv(self):
        raise NotImplementedError

    def reshape(self, *args):
        raise NotImplementedError

    def swapaxes(self, *args):
        raise NotImplementedError

    def mean(self, axis=None, keepdims=False):
        raise NotImplementedError

    def prod(self):
        raise NotImplementedError

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError


def array(array_like):
    return Array(np.array(array_like, dtype=config.dtype))


def asarray(array_like):
    if isinstance(array_like, Array):
        return array_like
    return Array(np.asarray(array_like, dtype=config.dtype))

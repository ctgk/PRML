import numpy as np
from prml.nn.config import config
from prml.nn.queue import backprop_queue


class Array(object):
    __array_ufunc__ = None

    def __init__(self, value, parent=None):
        self.value = np.atleast_1d(value)
        self.parent = parent
        self.grad = None
        self.gradtmp = None
        self.depth = 0 if parent is None else parent._out_depth()
        self.is_in_queue = False

    def __repr__(self):
        return f"Array(shape={self.value.shape}, dtype={self.value.dtype})"

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

    def backward(self, delta=None):
        if delta is None:
            delta = np.ones_like(self.value).astype(config.dtype)
        assert(delta.shape == self.value.shape)
        self._backward(delta)
        backprop_queue.enqueue(self)
        depth = self.depth
        while(len(backprop_queue)):
            queue = backprop_queue.dequeue(depth)
            if queue.parent is not None:
                queue.parent.backward(queue.gradtmp)
            queue.update_grad(queue.gradtmp)
            queue.gradtmp = None
            depth = queue.depth

    def update_grad(self, grad):
        if self.grad is None:
            self.grad = np.copy(grad)
        else:
            self.grad += grad

    def cleargrad(self):
        self.grad = None
        self.gradtmp = None

    def _backward(self, delta):
        if delta is None:
            return
        assert(delta.shape == self.shape)
        if self.gradtmp is None:
            self.gradtmp = np.copy(delta)
        else:
            self.gradtmp += delta

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

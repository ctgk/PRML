from prml.autodiff._core._function import _Function


class _Reshape(_Function):

    @staticmethod
    def _forward(x, shape):
        return x.reshape(*shape)

    @staticmethod
    def _backward(delta, x, shape):
        return delta.reshape(*x.shape)


def reshape(x, shape):
    """
    reshape array to specified shape

    Parameters
    ----------
    x : array_like
        input array to reshape
    shape : tuple
        target shape

    Returns
    -------
    Array
        reshaped Array
    """
    return _Reshape().forward(x, shape=shape)


def _reshape_method(x, *shape):
    return _Reshape().forward(x, shape=shape)

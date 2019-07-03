from prml.autodiff._core._function import _Function


class _Add(_Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(*args):
        return sum(args)

    @staticmethod
    def _backward(delta, *args):
        return tuple(delta for arg in args)


def add(*args):
    """
    element-wise addition arrays supporting automatic broadcasting

    .. math::

        x + y

    Parameters
    ----------
    args : array_like
        addend

    Returns
    -------
    Array
        summation of arrays
    """
    return _Add().forward(*args)

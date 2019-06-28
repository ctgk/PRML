from prml.autodiff._core._function import _Function


class _Add(_Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x + y

    @staticmethod
    def _backward(delta, x, y):
        return delta, delta


def add(x, y):
    """
    element-wise addition two arrays supporting automatic broadcasting

    .. math::

        x + y

    Parameters
    ----------
    x : array_like
        first addend
    y : array_like
        second addend

    Returns
    -------
    Array
        summation of two arrays
    """
    return _Add().forward(x, y)

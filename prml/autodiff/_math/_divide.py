from prml.autodiff._core._function import _Function


class _Divide(_Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x / y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta / y
        dy = -delta * x / (y ** 2)
        return dx, dy


def divide(x, y):
    """
    element-wise division of two arrays supporting automatic broadcasting

    .. math::

        x / y

    Parameters
    ----------
    x : array_like
        dividend
    y : array_like
        divisor

    Returns
    -------
    Array
        quotient
    """
    return _Divide().forward(x, y)


def _rdivide(x, y):
    return _Divide().forward(y, x)

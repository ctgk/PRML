from prml.autodiff.core.function import Function


class Add(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x + y

    @staticmethod
    def _backward(delta, x, y):
        return delta, delta


def add(x, y):
    """
    add two arrays

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
    return Add().forward(x, y)

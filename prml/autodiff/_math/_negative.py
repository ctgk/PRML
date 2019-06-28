from prml.autodiff._core._function import _Function


class Negative(_Function):

    @staticmethod
    def _forward(x):
        return -x

    @staticmethod
    def _backward(delta, x):
        return -delta


def negative(x):
    return Negative().forward(x)

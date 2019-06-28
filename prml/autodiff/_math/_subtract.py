from prml.autodiff._core._function import _Function


class Subtract(_Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x - y

    @staticmethod
    def _backward(delta, x, y):
        return delta, -delta


def subtract(x, y):
    return Subtract().forward(x, y)


def rsubtract(x, y):
    return Subtract().forward(y, x)

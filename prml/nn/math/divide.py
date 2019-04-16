from prml.nn.function import Function


class Divide(Function):
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
    return Divide().forward(x, y)


def rdivide(x, y):
    return Divide().forward(y, x)

from prml.nn.function import Function


class Subtract(Function):
    """Subtraction function."""

    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        return x - y

    @staticmethod
    def _backward(delta, x, y):
        return delta, -delta


def subtract(x, y):
    """Subtract."""
    return Subtract().forward(x, y)


def rsubtract(x, y):
    """Reverse subtract."""
    return Subtract().forward(y, x)

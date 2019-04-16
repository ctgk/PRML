from prml.nn.function import Function


class Negative(Function):

    @staticmethod
    def _forward(x):
        return -x

    @staticmethod
    def _backward(delta, x):
        return -delta


def negative(x):
    return Negative().forward(x)

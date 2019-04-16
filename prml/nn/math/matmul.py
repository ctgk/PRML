from prml.nn.function import Function


class Matmul(Function):

    @staticmethod
    def _forward(x, y):
        return x @ y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta @ y.T
        dy = x.T @ delta
        return dx, dy


def matmul(x, y):
    return Matmul().forward(x, y)


def rmatmul(x, y):
    return Matmul().forward(y, x)

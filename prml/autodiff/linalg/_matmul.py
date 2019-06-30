from prml.autodiff._core._function import _Function


class _Matmul(_Function):

    @staticmethod
    def _forward(x, y):
        return x @ y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta @ y.T
        dy = x.T @ delta
        return dx, dy


def matmul(x, y):
    return _Matmul().forward(x, y)


def rmatmul(x, y):
    return _Matmul().forward(y, x)

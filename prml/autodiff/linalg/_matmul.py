from itertools import zip_longest

from prml.autodiff._core._function import broadcast_to
from prml.autodiff.linalg._function import _LinAlgFunction


class _Matmul(_LinAlgFunction):
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(x, y):
        shape = []
        for i, (lenx, leny) in enumerate(
            zip_longest(reversed(x.shape), reversed(y.shape), fillvalue=1)
        ):
            if i > 1:
                shape.insert(0, max(lenx, leny))
        output = []
        output.append(broadcast_to(x, shape + list(x.shape[-2:])))
        output.append(broadcast_to(y, shape + list(y.shape[-2:])))
        return output

    @staticmethod
    def _forward(x, y):
        return x @ y

    @classmethod
    def _backward(cls, delta, x, y):
        dx = delta @ cls._T(y)
        dy = cls._T(x) @ delta
        return dx, dy


def matmul(x, y):
    return _Matmul().forward(x, y)


def rmatmul(x, y):
    return _Matmul().forward(y, x)

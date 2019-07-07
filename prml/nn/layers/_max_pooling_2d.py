from prml.autodiff.signal._max_pooling_2d import _MaxPooling2d
from prml.nn.layers._layer import _Layer


class MaxPooling2d(_Layer):

    def __init__(
        self,
        pool_size: int or tuple = (2, 2),
        stride: int or tuple = (2, 2),
        pad: int or tuple = (0, 0),
        activation=None
    ):
        super().__init__(activation, bias=None)
        if isinstance(pool_size, int):
            pool_size = (pool_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(pad, int):
            pad = (pad,) * 2
        self._func = _MaxPooling2d(pool_size, stride, pad)

    def _forward(self, x):
        return self._func.forward(x)

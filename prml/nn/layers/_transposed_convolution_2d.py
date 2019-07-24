from prml.autodiff.signal._transposed_convolution_2d import (
    _TransposedConvolution2d)
from prml.nn.layers._layer import _Layer


class TransposedConvolution2d(_Layer):

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        kernel_size: int or tuple = (3, 3),
        stride: int or tuple = (1, 1),
        pad: int or tuple = (0, 0),
        activation=None,
        initializer=None,
        has_bias: bool = True
    ):
        super().__init__(activation, initializer)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(pad, int):
            pad = (pad,) * 2
        with self.initialize():
            self.kernel_flat = self.initializer(size=(
                kernel_size[0] * kernel_size[1] * channel_out, channel_in
            ))
        if has_bias:
            self.initialize_bias(channel_out)
        self._func = _TransposedConvolution2d(kernel_size, stride, pad)

    def _forward(self, x):
        return self._func.forward(x, self.kernel_flat)

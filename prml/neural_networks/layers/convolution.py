import numpy as np
from scipy.stats import truncnorm
from .layer import Layer
from .util import img2patch, patch2img


class Convolution2d(Layer):
    """Convolution"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, std=1., istrainable=True):
        """
        construct convolution layer

        Parameters
        ----------
        in_channels : int
            number of channels of input
        out_channels : int
            number of channels of output
        ksize : int or tuple
            size of kernels
        stride : int or tuple
            stride of kernel applications
        pad : int or tuple
            padding image
        std : float
            standard deviation of truncated normal distribution for initializing parameter
        istrainable : bool
            flag indicating trainable or not

        Returns
        -------
        param : (ksize, ksize, dim_in, dim_out) ndarray
            convolution kernel
        deriv : (ksize, ksize, dim_in, dim_out) ndarray
            derivative of a cost function with respect to the kernel
        """
        if isinstance(ksize, int):
            self.ksize = (ksize,) * 2
        else:
            self.ksize = ksize
        if isinstance(stride, int):
            self.stride = (stride,) * 2
        else:
            self.stride = stride
        if isinstance(pad, int):
            self.pad = (0,) + (pad,) * 2 + (0,)
        else:
            self.pad = (0,) + tuple(pad) + (0,)
        self.param = np.float32(truncnorm(
            a=-2 * std, b=2 * std, scale=std).rvs(
                self.ksize + (in_channels, out_channels)))
        self.deriv = np.zeros_like(self.param)
        self.istrainable = istrainable

    def forward(self, x, training=False):
        """
        apply kernel convolution

        Parameters
        ----------
        x : (n_batch, xlen_in, ylen_in, dim_in) ndarray
            input image

        Returns
        -------
        output : (n_batch, xlen_out, ylen_out, dim_out) ndarray
            convoluted image
            len_out = (len_in - ksize) // stride + 1
        """
        x = np.pad(x, [(p,) for p in self.pad], "constant")
        if training:
            self.shape = x.shape
            self.patch = img2patch(x, self.ksize, self.stride)
            return np.tensordot(self.patch, self.param, axes=((3, 4, 5), (0, 1, 2)))
        else:
            return np.tensordot(
                img2patch(x, self.ksize, self.stride),
                self.param,
                axes=((3, 4, 5), (0, 1, 2)))

    def backward(self, delta):
        """
        backpropagate output error

        Parameters
        ----------
        delta : (n_batch, xlen_out, ylen_out, dim_out) ndarray
            output error

        Returns
        -------
        delta_in : (n_batch, xlen_in, ylen_in, dim_in) ndarray
            input error
            len_out = (len_in - ksize) // stride + 1
        """
        delta_in = patch2img(
            np.tensordot(delta, self.param, (3, 3)),
            self.stride,
            self.shape)
        slices = [slice(p, len_ - p) for p, len_ in zip(self.pad, self.shape)]
        delta_in = delta_in[slices]
        if self.istrainable:
            self.deriv = np.tensordot(self.patch, delta, axes=((0, 1, 2),) * 2)
        return delta_in

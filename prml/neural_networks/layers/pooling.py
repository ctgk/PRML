import numpy as np
from .layer import Layer
from .util import img2patch, patch2img


class MaxPooling2d(Layer):
    """MaxPooling2d"""

    def __init__(self, ksize, stride=1, pad=0):
        """
        construct max pooling layer

        Parameters
        ----------
        ksize : int
            window size
        stride : int
            stride of window applications
        pad : int
            pad width
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
        self.istrainable = False

    def forward(self, x, training=False):
        """
        spatial max pooling

        Parameters
        ----------
        x : (n_batch, xlen_in, ylen_in, in_channels) ndarray
            input image

        Returns
        -------
        output : (n_batch, xlen_out, ylen_out, in_channels) ndarray
            max pooled image
            len_out = (len_in - ksize) // stride + 1
        """
        x = np.pad(x, [(p,) for p in self.pad], "constant")
        patch = img2patch(x, self.ksize, self.stride)
        n_batch, xlen_out, ylen_out, _, _, in_channels = patch.shape
        patch = patch.reshape(n_batch, xlen_out, ylen_out, -1, in_channels)
        if training:
            self.shape = x.shape
            self.index = patch.argmax(axis=3)
        output = patch.max(axis=3)
        return output

    def backward(self, delta):
        """
        backpropagate output error

        Parameters
        ----------
        delta : (n_batch, xlen_out, ylen_out, in_channels) ndarray
            output error

        Returns
        -------
        delta_in : (n_batch, xlen_in, ylen_in, in_channels) ndarray
            input error
            len_out = (len_in - ksize) // stride + 1
        """
        delta_patch = np.zeros(delta.shape + (self.ksize[0] * self.ksize[1],), dtype=delta.dtype)
        index = np.where(delta == delta) + (self.index.ravel(),)
        delta_patch[index] = delta.ravel()
        delta_patch = np.reshape(delta_patch, delta.shape + self.ksize)
        delta_patch = delta_patch.transpose(0, 1, 2, 4, 5, 3)
        delta_in = patch2img(delta_patch, self.stride, self.shape)
        slices = [slice(p, len_ - p) for p, len_ in zip(self.pad, self.shape)]
        delta_in = delta_in[slices]
        return delta_in

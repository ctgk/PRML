import numpy as np

from prml.autodiff._core._function import _Function
from prml.autodiff.signal._util import patch2img, img2patch


class _TransposedConvolution2d(_Function):

    def __init__(self, kernel_shape, out_ch, stride, pad, shape):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.out_ch = out_ch
        self.pad = (0,) + pad + (0,)
        self.shape = shape

    def _forward(self, img, kernel_flat):
        if self.shape is None:
            shape = (len(img),) + tuple(
                s * (imlen - 1) + klen for s, imlen, klen
                in zip(self.stride, img.shape[1:], self.kernel_shape)
            ) + (self.out_ch,)
        else:
            shape = (len(img),) + self.shape + (self.out_ch,)
        patch_flat = np.matmul(img, kernel_flat.T)
        output = patch2img(patch_flat.reshape(
            *patch_flat.shape[:3], *self.kernel_shape, -1), self.stride, shape)
        output = output[
            :,
            self.pad[1]: output.shape[1] - self.pad[1],
            self.pad[2]: output.shape[2] - self.pad[2]
        ]
        return output

    def _backward(self, delta, img, kernel_flat):
        delta = np.pad(delta, [(p,) for p in self.pad], "constant")
        dpatch = img2patch(delta, self.kernel_shape, self.stride)
        dpatch_flat = dpatch.reshape(-1, kernel_flat.shape[0])
        dimg = np.matmul(dpatch_flat, kernel_flat).reshape(img.shape)
        dkernel_flat = np.matmul(
            img.reshape(-1, img.shape[-1]).T, dpatch_flat).T
        return dimg, dkernel_flat


def transposed_convolution_2d(img, kernel, stride=1, pad=0, shape=None):
    """
    transposed convolution

    Parameters
    ----------
    img : array_like (n_batch, xlen, ylen, in_channel)
        input to be convolved
    kernel : array_like (kx, ky, out_channel, in_channel)
        transposed kernel
    stride : int or tuple of ints (sx, sy)
        stride of kernel application
    pad : int or tuple of ints (px, py)
        padding image
    shape : tuple of ints (xlen', ylen')
        desired shape of output image
        If not specified, the output has the following length
        len' = s * (len - 1) - 2p + k

    Returns
    -------
    output : array_like (n_batch, xlen', ylen', out_channel)
        The first argument convoled with the second one
        len' will be the following if not specified
        len' = s * (len - 1) - 2p + k
    """
    trans_conv = _TransposedConvolution2d(
        kernel.shape[:2], kernel.shape[2], stride, pad, shape)
    return trans_conv.forward(img, kernel.reshape(-1, kernel.shape[-1]))

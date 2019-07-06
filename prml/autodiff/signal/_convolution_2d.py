import numpy as np

from prml.autodiff._core._function import _Function
from prml.autodiff.signal._util import img2patch, patch2img


class _Convolution2d(_Function):

    def __init__(self, kernel_shape, stride, pad):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = (0,) + pad + (0,)

    def _forward(self, img, kernel_flat):
        img_padded = np.pad(img, [(p,) for p in self.pad], "constant")
        self.shape_padded = img_padded.shape
        self.patch = img2patch(img_padded, self.kernel_shape, self.stride)
        self.shape_out = self.patch.shape[:3] + (kernel_flat.shape[1],)
        self.patch_flat = self.patch.reshape(-1, kernel_flat.shape[0])
        return np.matmul(self.patch_flat, kernel_flat).reshape(self.shape_out)

    def _backward(self, delta, img, kernel_flat):
        delta_flat = delta.reshape(-1, delta.shape[-1])
        dpatch_flat = delta_flat @ kernel_flat.T
        dpatch = dpatch_flat.reshape(self.patch.shape)
        dimg_padded = patch2img(dpatch, self.stride, self.shape_padded)
        slices = tuple(
            slice(p, len_ - p) for p, len_
            in zip(self.pad, self.shape_padded)
        )
        dimg = dimg_padded[slices]
        dkernel_flat = self.patch_flat.T @ delta_flat
        return dimg, dkernel_flat


def convolution_2d(img, kernel, stride=(1, 1), pad=(0, 0)):
    """
    returns convolution of two tensors

    Parameters
    ----------
    img : array_like (n_batch, xlen, ylen, in_channel)
        input image to be convolved
    kernel : array_like (kx, ky, in_channel, out_channel)
        convolution kernel
    stride : tuple of ints (sx, sy)
        stride of kernel application
    pad : tuple of ints (px, py)
        padding image

    Returns
    -------
    output : array_like (n_batch, xlen', ylen', out_channel)
        input convolved with kernel
        len' = (len + 2p - k) // s + 1
    """
    conv = _Convolution2d(kernel.shape[:2], stride, pad)
    return conv.forward(img, kernel.reshape(-1, kernel.shape[-1]))

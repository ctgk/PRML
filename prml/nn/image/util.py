import itertools
import numpy as np
from numpy.lib.stride_tricks import as_strided


def img2patch(img, size, step=1):
    """
    convert batch of image array into patches
    Parameters
    ----------
    img : (n_batch, xlen_in, ylen_in, in_chaprml.nnels) ndarray
        batch of images
    size : tuple or int
        patch size
    step : tuple or int
        stride of patches
    Returns
    -------
    patch : (n_batch, xlen_out, ylen_out, size, size, in_chaprml.nnels) ndarray
        batch of patches at all points
        len_out = (len_in - size) // step + 1
    """
    ndim = img.ndim
    if isinstance(size, int):
        size = (size,) * (ndim - 2)
    if isinstance(step, int):
        step = (step,) * (ndim - 2)

    slices = tuple(slice(None, None, s) for s in step)
    window_strides = img.strides[1:]
    index_strides = img[(slice(None),) + slices].strides[:-1]

    out_shape = tuple(
        np.subtract(img.shape[1: -1], size) // np.array(step) + 1)
    out_shape = (len(img),) + out_shape + size + (np.size(img, -1),)
    strides = index_strides + window_strides
    patch = as_strided(img, shape=out_shape, strides=strides)
    return patch


def _patch2img(x, stride, shape):
    """
    sum up patches and form an image
    Parameters
    ----------
    x : (n_batch, xlen_in, ylen_in, kx, ky, in_chaprml.nnels) ndarray
        batch of patches at all points
    stride : tuple
        applied stride to take patches
    shape : (n_batch, xlen_out, ylen_out, in_chaprml.nnels) tuple
        output image shape
    Returns
    -------
    img : (n_batch, xlen_out, ylen_out, in_chaprml.nnels) ndarray
        image
    """
    img = np.zeros(shape, dtype=x.dtype)
    kx, ky = x.shape[3: 5]
    for i, j in itertools.product(range(kx), range(ky)):
        slices = tuple(slice(b, b + s * len_, s) for b, s, len_ in zip([i, j], stride, x.shape[1: 3]))
        img[(slice(None),) + slices] += x[..., i, j, :]
    return img


def patch2img(x, stride, shape):
    img = np.zeros(shape, dtype=x.dtype)
    kx, ky = x.shape[3:5]
    patch = img2patch(img, (kx, ky), stride)
    for i, j in itertools.product(range(kx), range(ky)):
        patch[..., i, j, :] += x[..., i, j, :]
    return img


def patch2img_no_overlap(x, stride, shape):
    img = np.zeros(shape, dtype=x.dtype)
    patch = img2patch(img, x.shape[3:5], stride)
    patch += x
    return img

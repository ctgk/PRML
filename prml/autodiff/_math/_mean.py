from prml.autodiff._math._sum import sum


def mean(x, axis=None, keepdims=False):
    """
    returns arithmetic mean of the elements along given axis

    Parameters
    ----------
    x : array_like
        input array
    axis : int, tuple, optional
        axis along which mean is computed, by default None
    keepdims : bool, optional
        flag to keep the dimension of an input array, by default False

    Returns
    -------
    Array
        arithmetic mean of an array

    Raises
    ------
    TypeError
        throws an error if axis is neither int nor tuples
    """
    if axis is None:
        return sum(x, axis=None, keepdims=keepdims) / x.size
    elif isinstance(axis, int):
        N = x.shape[axis]
        return sum(x, axis=axis, keepdims=keepdims) / N
    elif isinstance(axis, tuple):
        N = 1
        for ax in axis:
            N *= x.shape[ax]
        return sum(x, axis=axis, keepdims=keepdims) / N
    else:
        raise TypeError(f"Unsupported type for axis: {type(axis)}")

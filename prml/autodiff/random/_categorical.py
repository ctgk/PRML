import numpy as np

from prml.autodiff._core._function import _Function, broadcast_to
from prml.autodiff._math._log import log
from prml.autodiff._nonlinear._log_softmax import log_softmax
from prml.autodiff._nonlinear._softmax import softmax


class _Categorical(_Function):

    def _forward(mean):
        if mean.ndim == 1:
            index = np.random.choice(mean.shape[-1], p=mean)
            return np.eye(mean.shape[-1])[index]
        else:
            indices = [
                np.random.choice(m.size, p=m)
                for m in mean.reshape(-1, mean.shape[-1])]
            onehot = np.eye(mean.shape[-1])[np.array(indices)]
            return onehot.reshape(*mean.shape)


def categorical(mean=None, logit=None, temperature: float = None, size=None):
    """
    sample from categorical distribution

    Parameters
    ----------
    mean : array_like, optional
        mean of the distribution to sample from, by default None
    logit : array_like, optional
        logit of the distribution to sample from, by default None
    temperature : float, optional
        use gumbel-softmax if specified, by default None
    size : int, tuple, optional
        size of sample, by default None

    Returns
    -------
    Array
        one-of-k coded sample from the distribution

    Raises
    ------
    ValueError
        raises if both mean and logit are passed or neither of them are
    """
    if ((mean is None and logit is None)
            or (mean is not None and logit is not None)):
        raise ValueError("Pass either mean or logit. Not both or neither.")

    if temperature is not None:
        if mean is not None:
            logit = log(mean)
        if size is not None:
            logit = broadcast_to(logit, size)
        g = np.random.gumbel(size=logit.shape)
        return softmax((logit + g) / temperature)

    if mean is None:
        mean = softmax(logit)
    if size is not None:
        mean = broadcast_to(mean, size)
    return _Categorical().forward(mean)


def softmax_cross_entropy(x, logit):
    return -x * log_softmax(logit)


def categorical_logpdf(x, logit):
    return x * log_softmax(logit)

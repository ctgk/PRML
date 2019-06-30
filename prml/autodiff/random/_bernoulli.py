import numpy as np

from prml.autodiff._core._function import _Function, broadcast_to
from prml.autodiff._nonlinear._logit import logit as logit_function
from prml.autodiff._nonlinear._sigmoid import sigmoid


class _Bernoulli(_Function):

    def _forward(mean):
        return np.random.uniform(size=mean.shape) < mean

    def _backward(self, delta, mean):
        raise NotImplementedError


def bernoulli(mean=None, logit=None, temperature: float = None, size=None):
    if ((mean is None and logit is None)
            or (mean is not None and logit is not None)):
        raise ValueError("Pass either mean or logit. Not both or neither.")

    if temperature is not None:
        if mean is not None:
            logit = logit_function(mean)
        if size is not None:
            logit = broadcast_to(logit, size)
        g = (np.random.gumbel(size=logit.shape)
             - np.random.gumbel(size=logit.shape))
        return sigmoid((logit + g) / temperature)

    if mean is None:
        mean = sigmoid(logit)
    if size is not None:
        mean = broadcast_to(mean, size)
    return _Bernoulli().forward(mean)

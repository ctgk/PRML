import numpy as np
from scipy.stats import truncnorm

from prml.nn.array.array import asarray


def normal(mean, std, size):
    """Return a random sample from normal distribution."""
    return asarray(np.random.normal(mean, std, size))


def truncnormal(min_, max_, scale, size):
    """Return a random sample from trunc-normal distribution."""
    return asarray(truncnorm(a=min_, b=max_, scale=scale).rvs(size))

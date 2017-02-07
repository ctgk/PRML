import random
import numpy as np


def metropolis_hastings(func, dist, n, downsample=1):
    """
    Metropolis Hastings algorith

    Parameters
    ----------
    func : callable
        (un)normalized distribution to be sampled from
    dist
        proposal distribution
    n : int
        number of samples to draw
    downsample : int
        downsampling factor

    Returns
    -------
    sample : (n, ndim) ndarray
        generated sample
    """
    x = np.zeros((1, dist.ndim))
    sample = []
    for i in range(n * downsample):
        x_new = x + dist.draw()
        accept_proba = func(x_new) * dist.proba(x - x_new) / (func(x) * dist.proba(x_new - x))
        if random.random() < accept_proba:
            x = x_new
        if i % downsample == 0:
            sample.append(x[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, dist.ndim), sample.shape
    return sample

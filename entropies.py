import numpy as np


def entropy(p):
    """
    entropy of a random variable
    H = -sum_x p(x)log{p(x)}

    Parameters
    ----------
    p : ndarray
        the distribution of the random variable

    Returns
    -------
    H : float
        entropy
    """
    return -np.sum(p * np.log(p))


def kl_divergence(p, q):
    """
    Kullback-Leibler divergence also known as relative entropy
    KL(p||q) = sum_x p(x) log{p(x) / q(x)}

    Parameters
    ----------
    p : ndarray
        a probability distribution
    q : ndarray
        another probabilitiy distribution

    Returns
    -------
    KL : float
        Kullback-Leibler divergence
    """
    return np.sum(p * np.log(p) - p * np.log(q))


def mutual_information(p):
    """
    Mutual information between two random variables
    I[x,y] = KL(p(x,y)||p(x)p(y))

    Parameters
    ----------
    p : ndarray [xlen, ylen]
        joint probability distribution of two random variables

    Returns
    -------
    I : float
        Mutual information
    """
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    return kl_divergence(p, px[:, None] * py)

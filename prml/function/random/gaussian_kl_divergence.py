from prml.function.math import log, square


def gaussian_kl_divergence(m1, s1, m2, s2):
    """
    KL divergence of two gaussian distributions

    Parameters
    ----------
    m1
        mean parameter of first gaussian distribution
    s1
        standard deviation of first gaussian distribution
    m2
        mean of second gaussian distribution
    s2
        standard deviation of second gaussian distribution

    Returns
    -------
    output : Tensor
        KL divergence from first gaussian to second one
    """
    return log(s2) - log(s1) + 0.5 * (square(s1) + square(m1 - m2)) / square(s2) - 0.5
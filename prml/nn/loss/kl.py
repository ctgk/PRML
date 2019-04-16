import numpy as np
from prml.nn.function import Function
from prml.nn.distribution.bernoulli import Bernoulli
from prml.nn.distribution.categorical import Categorical
from prml.nn.distribution.gaussian import Gaussian
from prml.nn.math.log import log
from prml.nn.math.square import square
from prml.nn.nonlinear.log_softmax import log_softmax
from prml.nn.nonlinear.softplus import softplus


def kl_divergence(q, p, data=None):
    """
    compute sample approximation of kl divergence from p to q
    KL(q||p)

    Parameters
    ----------
    q : Distribution
        one generated a sample
    p : Distribution
    data : Array
    """
    if isinstance(q, Bernoulli) and isinstance(p, Bernoulli):
        return kl_bernoulli(q, p)
    elif isinstance(q, Categorical) and isinstance(p, Categorical):
        return kl_categorical(q, p)
    elif isinstance(q, Gaussian) and isinstance(p, Gaussian):
        return kl_gaussian(q, p)
    elif q.data.depth > 0:
        return q.log_pdf() - p.log_pdf(q.data)
    else:
        return q.pdf() * (q.log_pdf() - p.log_pdf(q.data))


def kl_bernoulli(q, p):
    return (q.mean - 1) * (q.logit - p.logit) \
        - softplus(-q.logit) + softplus(p.logit)


def kl_categorical(q, p):
    return (q.mean * (log_softmax(q.logit) - log_softmax(p.logit))).sum(axis=-1)


def kl_gaussian(q, p):
    qvar = square(q.std)
    pvar = square(p.std)
    return log(p.std) - log(q.std) + 0.5 * (qvar + square(p.mean - q.mean)) / pvar - 0.5



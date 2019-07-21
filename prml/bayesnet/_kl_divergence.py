from prml import autodiff
from prml.bayesnet.functions._gaussian import Gaussian


def kl_divergence(p, q, **kwargs):
    if isinstance(p, Gaussian) and isinstance(q, Gaussian):
        return kl_gaussian(p, q, **kwargs)
    sample = p.sample()
    sample.update(kwargs)
    return p.log_pdf(**sample) - q.log_pdf(**sample)


def kl_gaussian(p, q, **kwargs):
    p_ = p.forward(**{key: kwargs[key] for key in p.conditions})
    pmean, pstd = p_["mean"], p_["std"]
    pvar = autodiff.square(pstd)
    q_ = q.forward(**{key: kwargs[key] for key in q.conditions})
    qmean, qstd = q_["mean"], q_["std"]
    qvar = autodiff.square(qstd)
    return (
        autodiff.log(qstd) - autodiff.log(pstd)
        + 0.5 * (pvar + autodiff.square(pmean - qmean)) / qvar - 0.5
    ).sum()


def kl_exponential(p, q, **kwargs):
    prate = p.forward(**{key: kwargs[key] for key in p.conditions})["rate"]
    qrate = q.forward(**{key: kwargs[key] for key in q.conditions})["rate"]
    rate_of_rate = qrate / prate
    return (rate_of_rate - autodiff.log(rate_of_rate) - 1).sum()

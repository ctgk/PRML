from prml import autodiff
from prml.bayesnet.functions._gaussian import Gaussian


def kl_divergence(p, q, **kwargs):
    if isinstance(p, Gaussian) and isinstance(q, Gaussian):
        return kl_gaussian(p, q, **kwargs)
    sample = p.sample()
    sample.update(kwargs)
    return p.log_pdf(**sample) - q.log_pdf(**sample)


def kl_gaussian(p, q, **kwargs):
    p_ = p.forward(**kwargs)
    pmean, pstd = p_["mean"], p_["std"]
    pvar = autodiff.square(pstd)
    q_ = q.forward(**kwargs)
    qmean, qstd = q_["mean"], q_["std"]
    qvar = autodiff.square(qstd)
    return (
        autodiff.log(qstd) - autodiff.log(pstd)
        + 0.5 * (pvar + autodiff.square(pmean - qmean)) / qvar - 0.5
    ).sum()

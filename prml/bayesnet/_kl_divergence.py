def kl_divergence(p, q, **kwargs):
    sample = p.sample()
    sample.update(kwargs)
    return p.log_pdf(**sample) - q.log_pdf(**sample)

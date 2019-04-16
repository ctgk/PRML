import numpy as np
from prml.nn.array.array import asarray
from prml.nn.math.log import log
from prml.nn.nonlinear.sigmoid import sigmoid
from prml.nn.nonlinear.logit import logit as logit_func
from prml.nn.distribution.distribution import Distribution
from prml.nn.loss.sigmoid_cross_entropy import sigmoid_cross_entropy


class Bernoulli(Distribution):
    is_categorical = True

    def __init__(self, mean=None, logit=None):
        super().__init__()
        if mean is not None:
            self.mean = asarray(mean)
            assert((self.mean.value >= 0).all() and (self.mean.value <= 1).all())
            self.logit = logit_func(mean)
            self._log_pdf = self._log_pdf_mu
        elif logit is not None:
            self.mean = sigmoid(logit)
            self.logit = asarray(logit)
            self._log_pdf = self._log_pdf_logit
        else:
            raise ValueError

    def forward(self):
        binary = (np.random.uniform(size=self.mean.shape) < self.mean.value)
        return asarray(binary)

    def _pdf(self, x):
        return (self.mean ** x) * (1 - self.mean) ** (1 - x)

    def _log_pdf_mu(self, x):
        return x * log(self.mean) + (1 - x) * log(1 - self.mean)

    def _log_pdf_logit(self, x):
        return -sigmoid_cross_entropy(self.logit, x)

import numpy as np
from prml.nn.array.array import asarray
from prml.nn.function import Function
from prml.nn.math.log import log
from prml.nn.nonlinear.softmax import softmax
from prml.nn.distribution.distribution import Distribution
from prml.nn.loss.softmax_cross_entropy import softmax_cross_entropy


class Categorical(Distribution):
    is_categorical = True

    def __init__(self, mean=None, logit=None, use_gumbel_softmax=True, tau=0.1):
        super().__init__()
        if mean is not None:
            self.mean = asarray(mean)
            assert((self.mean.value >= 0).all() and np.allclose(self.mean.value.sum(axis=-1), 1))
            self.logit = log(self.mean)
            self._log_pdf = self._log_pdf_mean
        elif logit is not None:
            self.mean = softmax(logit)
            self.logit = asarray(logit)
            self._log_pdf = self._log_pdf_logit
        else:
            raise ValueError
        self.n_category = self.mean.shape[-1]
        if use_gumbel_softmax:
            self.forward = self._forward_gumbel_softmax
        else:
            self.forward = self._forward
        self.tau = tau
        self.eye = np.eye(self.n_category)

    def _forward_gumbel_softmax(self):
        g = np.random.gumbel(size=self.mean.shape)
        return softmax((self.logit + g) / self.tau)

    def _forward(self):
        if self.mean.ndim == 1:
            index = np.random.choice(self.n_category, p=self.mean.value)
            return asarray(self.eye[index])
        else:
            mean = self.mean.value.reshape(-1, self.n_category)
            indices = [np.random.choice(self.n_category, p=p) for p in mean]
            onehot = self.eye[np.array(indices)]
            return asarray(onehot)

    def _pdf(self, x):
        return CategoricalPDF().forward(self.mean, x)

    def _log_pdf_mean(self, x):
        return x * log(self.mean)

    def _log_pdf_logit(self, x):
        return -softmax_cross_entropy(self.logit, x)


class CategoricalPDF(Function):

    def _forward(self, mean, x):
        proba = np.ones_like(mean)
        self.indices = np.where(x == 1)
        proba[self.indices] = mean[self.indices]
        return proba

    def _backward(self, delta, mean, x):
        dmean = np.zeros_like(mean)
        dmean[self.indices] = delta[self.indices]
        return dmean

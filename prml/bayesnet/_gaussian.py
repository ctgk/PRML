from typing import Iterable

import numpy as np

from prml import autodiff
from prml.bayesnet._distribution import Distribution


class Gaussian(Distribution):

    def __init__(self, out: Iterable[str], mean=0, std=1,
                 conditions: Iterable[str] = [], name: str = "N"):
        if len(out) != 1:
            raise ValueError
        super().__init__(out, conditions, name)
        self.mean = mean
        self.std = std

    def forward(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def log_pdf(self, **kwargs):
        return autodiff.random.gaussian_logpdf(
            kwargs[self.out[0]],
            **self.forward(**{key: kwargs[key] for key in self.conditions})
        ).sum()

    def sample(self, size=None) -> dict:
        sample = autodiff.random.gaussian(**self.forward(), size=size)
        return {self.out[0]: sample}


class MultivariateGaussian(Distribution):

    def __init__(
        self,
        out: Iterable[str],
        mean=np.zeros(2),
        covariance=np.eye(2),
        conditions: Iterable[str] = [],
        name: str = "N"
    ):
        if len(out) != 1:
            raise ValueError
        super().__init__(out, conditions, name)
        self.mean = mean
        self.covariance

    def forward(self) -> dict:
        return {"mean": self.mean, "covariance": self.covariance}

    def log_pdf(self, **kwargs):
        raise NotImplementedError

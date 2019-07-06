from typing import Iterable

from prml import autodiff
from prml.bayesnet._distribution import Distribution


class Bernoulli(Distribution):

    def __init__(
        self,
        out: Iterable[str],
        mean=0.5,
        conditions: Iterable[str] = [],
        name: str = "Bern"
    ):
        if len(out) != 1:
            raise ValueError
        super().__init__(out, conditions, name)
        self.mean = mean

    def forward(self) -> dict:
        return {"mean": self.mean}

    def log_pdf(self, **kwargs):
        x = kwargs[self.out[0]]
        param = self.forward(
            **{key: kwargs[key] for key in self.conditions})
        if "mean" in param:
            return (
                x * autodiff.log(param["mean"])
                + (1 - x) * autodiff.log(1 - param["mean"])
            ).sum()
        elif "logit" in param:
            return autodiff.random.bernoulli_logpdf(x, param["logit"]).sum()

    def pdf(self, **kwargs):
        return autodiff.exp(self.log_pdf(**kwargs))

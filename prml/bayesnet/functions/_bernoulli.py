from prml import autodiff
from prml.bayesnet.functions._function import ProbabilityFunction


class Bernoulli(ProbabilityFunction):

    def __init__(
        self,
        var: str or list,
        mean=0.5,
        conditions: list = [],
        name: str = "Bern"
    ):
        var = var if isinstance(var, list) else [var]
        if len(var) != 1:
            raise ValueError
        super().__init__(var, conditions, name)
        self._mean = autodiff.asarray(mean)
        self._mean.requires_grad = False

    def forward(self) -> dict:
        return {"mean": self._mean}

    def _log_pdf(self, **kwargs):
        x = kwargs[self.var[0]]
        param = self.forward(
            **{key: kwargs[key] for key in self.conditions})
        if "mean" in param:
            return (
                x * autodiff.log(param["mean"])
                + (1 - x) * autodiff.log(1 - param["mean"])
            ).sum()
        elif "logit" in param:
            return autodiff.random.bernoulli_logpdf(x, param["logit"]).sum()

    def _sample(self, temperature: float = None, **kwargs) -> dict:
        sample_ = autodiff.random.bernoulli(
            **self.forward(**kwargs), temperature=temperature)
        return {self.var[0]: sample_}

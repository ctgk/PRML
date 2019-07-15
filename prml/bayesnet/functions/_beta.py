from prml import autodiff
from prml.bayesnet.functions._function import ProbabilityFunction


class Beta(ProbabilityFunction):

    def __init__(
        self,
        var: str or list,
        a=1,
        b=1,
        conditions=[],
        name="Beta"
    ):
        var = var if isinstance(var, list) else [var]
        if len(var) != 1:
            raise ValueError
        super().__init__(var, conditions, name)
        self.a = a
        self.b = b

    def forward(self) -> dict:
        return {"a": self.a, "b": self.b}

    def log_pdf(self, **kwargs):
        x = kwargs[self.var[0]]
        return autodiff.random.beta_logpdf(
            x, **self.forward(**{key: kwargs[key] for key in self.conditions})
        ).sum()

    def _sample(self, **kwargs) -> dict:
        sample = autodiff.random.beta(**self.forward(**kwargs))
        return {self.var[0]: sample}

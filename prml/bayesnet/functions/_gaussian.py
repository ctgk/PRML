from prml import autodiff
from prml.bayesnet.functions._function import ProbabilityFunction


class Gaussian(ProbabilityFunction):

    def __init__(
        self,
        var: str or list = "x",
        conditions: list = [],
        name: str = "N",
        mean=0,
        std=1,
        size=None
    ):
        var = var if isinstance(var, list) else [var]
        if len(var) != 1:
            raise ValueError
        super().__init__(var, conditions, name)
        with self.initialize():
            if size is not None:
                self._mean = autodiff.zeros(size) + mean
                self._logstd = autodiff.zeros(size) + autodiff.log(std)
            else:
                self._mean = autodiff.asarray(mean)
                self._logstd = autodiff.log(std)

    def forward(self) -> dict:
        return {"mean": self._mean, "std": autodiff.exp(self._logstd)}

    def _log_pdf(self, **kwargs):
        return autodiff.random.gaussian_logpdf(
            kwargs[self.var[0]],
            **self.forward(**{key: kwargs[key] for key in self.conditions})
        ).sum()

    def _sample(self, **kwargs) -> dict:
        sample_ = autodiff.random.gaussian(**self.forward(**kwargs))
        return {self.var[0]: sample_}

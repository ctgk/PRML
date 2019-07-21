from prml import autodiff
from prml.bayesnet.functions._function import ProbabilityFunction


class Exponential(ProbabilityFunction):
    r"""
    Exponential distribution

    .. math::

        p(x|\lambda) = \lambda e^{-\lambda x}
    """

    def __init__(
        self,
        var: str = "x",
        conditions: list = [],
        name: str = "p",
        rate=1,
        size=None
    ):
        super().__init__(var=[var], conditions=conditions, name=name)
        with self.initialize():
            if size is not None:
                self._lograte = autodiff.zeros(size) + autodiff.log(rate)
            else:
                self._lograte = autodiff.log(rate)

    def forward(self) -> dict:
        return {"rate": autodiff.exp(self._lograte)}

    def _log_pdf(self, **kwargs):
        return autodiff.random.exponential_logpdf(
            kwargs[self.var[0]],
            **self.forward(**{key: kwargs[key] for key in self.conditions})
        ).sum()

    def _sample(self, **kwargs) -> dict:
        sample_ = autodiff.random.exponential(**self.forward(**kwargs))
        return {self.var[0]: sample_}

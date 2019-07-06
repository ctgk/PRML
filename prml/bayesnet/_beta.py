from prml import autodiff
from prml.bayesnet._distribution import Distribution


class Beta(Distribution):

    def __init__(
        self,
        out,
        a=1,
        b=1,
        conditions=[],
        name="Beta"
    ):
        if len(out) != 1:
            raise ValueError
        super().__init__(out, conditions, name)
        self.a = a
        self.b = b

    def forward(self) -> dict:
        return {"a": self.a, "b": self.b}

    def log_pdf(self, **kwargs):
        x = kwargs[self.out[0]]
        return autodiff.random.beta_logpdf(
            x, **self.forward(**{key: kwargs[key] for key in self.conditions})
        ).sum()

    def sample(self, size=None) -> dict:
        sample = autodiff.random.beta(**self.forward(), size=size)
        return {self.out[0]: sample}

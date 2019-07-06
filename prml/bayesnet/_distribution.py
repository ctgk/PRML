from typing import Iterable
from prml import autodiff


class Distribution(autodiff.Module):

    def __init__(
        self,
        out: Iterable[str],
        conditions: Iterable[str] = [],
        name: str = "p"
    ):
        """
        initialize distribution module

        Parameters
        ----------
        out : Iterable[str], optional
            name of output variables, by default None
        conditions : Iterable[str], optional
            name of conditional variables, by default None
        name : str, optional
            name of the distribution, by default None
        """
        super().__init__()
        self.out = out
        self.conditions = conditions
        self._formulation = f"{name}(" + ",".join(out)
        if conditions:
            self._formulation += "|" + ",".join(conditions)
        self._formulation += ")"
        self._factors = [self]

    def __repr__(self):
        return self._formulation

    def __mul__(self, other):
        if not isinstance(other, Distribution):
            raise TypeError(
                "Distribution only supports multiplication with Distribution")
        if set(self.out) & set(other.out):
            raise ValueError("invalid formuation of Distribution")
        out = set(self.out) | set(other.out)
        conditions = (set(self.conditions) | set(other.conditions)) - out
        out = Distribution(out=list(out), conditions=list(conditions))
        out._formulation = repr(self) + repr(other)
        out._factors = self._factors + other._factors
        return out

    def forward(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def log_pdf(self, **kwargs):
        if self in self._factors:
            raise NotImplementedError
        else:
            return autodiff.add(
                *tuple(factor.log_pdf(**kwargs) for factor in self._factors))

    def pdf(self, **kwargs):
        return autodiff.exp(self.log_pdf(**kwargs))

    def sample(self, *args, **kwargs) -> dict:
        raise NotImplementedError

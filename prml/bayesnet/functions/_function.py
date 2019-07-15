from typing import Iterable
from prml import autodiff


class ProbabilityFunction(autodiff.Module):
    """
    class for Probability density function and probability mass function

    Attributes
    ----------
    out : Iterable[str]
        name of output variables
    conditions : Iterable[str]
        name of conditional variables
    name : str
        name of the function
    """

    def __init__(
        self,
        var: Iterable[str],
        conditions: Iterable[str] = [],
        name: str = "p"
    ):
        """
        initialize distribution module

        Parameters
        ----------
        var : Iterable[str], optional
            name of output variables, by default None
        conditions : Iterable[str], optional
            name of conditional variables, by default None
        name : str, optional
            name of the function, by default None
        """
        super().__init__()
        self.var = var
        self.name = name
        self.conditions = conditions
        self._formulation = f"{name}(" + ",".join(var)
        if conditions:
            self._formulation += "|" + ",".join(conditions)
        self._formulation += ")"
        self._factors = [self]

    def __setattr__(self, key, value):
        if self.is_initializing:
            self.add_item(
                self.parameter, self.name + "".join(self.var) + key, value)
        object.__setattr__(self, key, value)

    def initialize_param(self, key, param):
        key = self.name + "".join(self.var) + key
        if param is None:
            object.__setattr__(self, key, None)
        else:
            with self.initialize():
                super().__setattr__(key, param)

    def __repr__(self):
        return self._formulation

    def __mul__(self, other):
        if not isinstance(other, ProbabilityFunction):
            raise TypeError(
                "ProbabilityFunction supports multiplication with "
                "ProbabilityFunction only.")
        if set(self.var) & set(other.var):
            raise ValueError("invalid formulation of ProbabilityFunction")
        var = set(self.var) | set(other.var)
        conditions = (set(self.conditions) - var) | set(other.conditions)
        if var & conditions:
            raise ValueError("invalid formulation of ProbabilityFunction")
        out = ProbabilityFunction(var=list(var), conditions=list(conditions))
        out._formulation = self._formulation + other._formulation
        out._factors = self._factors + other._factors
        out.parameter.update(self.parameter, **other.parameter)
        return out

    def forward(self, **kwargs) -> dict:
        raise NotImplementedError

    def _log_pdf(self, **kwargs):
        raise NotImplementedError

    def log_pdf(self, **kwargs):
        if (set(self.conditions) | set(self.var)) - set(kwargs):
            raise ValueError("Some variables are missing")
        if self in self._factors:
            return self._log_pdf(**kwargs)
        return autodiff.add(
            *tuple(factor.log_pdf(**kwargs) for factor in self._factors))

    def pdf(self, **kwargs):
        return autodiff.exp(self.log_pdf(**kwargs))

    def _sample(self, **kwargs) -> dict:
        raise NotImplementedError

    def sample(self, **kwargs) -> dict:
        """
        perform forward sampling also known as ancestral sampling
        given all conditional variables

        Returns
        -------
        dict
            sample for each random variable

        Raises
        ------
        NotImplementedError
        """
        if set(kwargs) ^ set(self.conditions):
            raise KeyError
        if self in self._factors:
            return self._sample(**kwargs)
        factors = [factor for factor in self._factors]
        sample_ = {}
        while len(factors):
            factor = factors.pop(0)
            if set(factor.conditions) - set(kwargs):
                factors.append(factor)
            else:
                sample_.update(factor.sample(
                    **{key: kwargs[key] for key in factor.conditions}))
                kwargs.update(sample_)
        return sample_

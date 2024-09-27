import numpy as np


class RandomVariable(object):
    """Base class for random variables."""

    def __init__(self):
        """Initialize a random variable."""
        self.parameter = {}

    def __repr__(self):
        """Representation of the random variable."""
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            if isinstance(value, RandomVariable):
                string += f"{key}={value:8}"
            else:
                string += f"{key}={value}"
            string += "\n"
        string += ")"
        return string

    def __format__(self, indent="4"):
        """Format the random variable."""
        indent = int(indent)
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * indent)
            if isinstance(value, RandomVariable):
                string += f"{key}=" + value.__format__(str(indent + 4))
            else:
                string += f"{key}={value}"
            string += "\n"
        string += (" " * (indent - 4)) + ")"
        return string

    def fit(self, x, **kwargs):
        """Estimate parameter(s) of the distribution.

        Parameters
        ----------
        x : np.ndarray
            observed data
        """
        self._check_input(x)
        if hasattr(self, "_fit"):
            self._fit(x, **kwargs)
        else:
            raise NotImplementedError

    # def ml(self, x, **kwargs):
    #     """
    #     maximum likelihood estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     x : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(x)
    #     if hasattr(self, "_ml"):
    #         self._ml(x, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def map(self, x, **kwargs):
    #     """
    #     maximum a posteriori estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     x : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(x)
    #     if hasattr(self, "_map"):
    #         self._map(x, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def bayes(self, x, **kwargs):
    #     """
    #     bayesian estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     x : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(x)
    #     if hasattr(self, "_bayes"):
    #         self._bayes(x, **kwargs)
    #     else:
    #         raise NotImplementedError

    def pdf(self, x):
        """Compute probability density function p(x|parameter).

        Parameters
        ----------
        x : (sample_size, ndim) np.ndarray
            input of the function

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        self._check_input(x)
        if hasattr(self, "_pdf"):
            return self._pdf(x)
        else:
            raise NotImplementedError

    def draw(self, sample_size=1):
        """Draw samples from the distribution.

        Parameters
        ----------
        sample_size : int
            sample size

        Returns
        -------
        sample : (sample_size, ndim) np.ndarray
            generated samples from the distribution
        """
        assert isinstance(sample_size, int)
        if hasattr(self, "_draw"):
            return self._draw(sample_size)
        else:
            raise NotImplementedError

    def _check_input(self, x):
        assert isinstance(x, np.ndarray)

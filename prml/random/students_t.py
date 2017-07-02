import numpy as np
from scipy.special import gamma, digamma
from prml.random.random import RandomVariable


class StudentsT(RandomVariable):
    """
    Student's t-distribution
    p(x|mu, tau(precision), dof)
    = (1 + tau * (x - mu)^2 / dof)^-(D + dof)/2 / const.
    """

    def __init__(self, mu=None, precision=None, dof=None):
        assert dof is None or isinstance(dof, (int, float))
        self.mu = mu
        self.precision = precision
        self.dof = dof

    def __setattr__(self, name, value):
        if name is "mu":
            self.set_mu(value)
        elif name is "precision":
            self.set_precision(value)
        else:
            object.__setattr__(self, name, value)

    def set_mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            object.__setattr__(self, "mu", np.array(mu))
        elif isinstance(mu, np.ndarray):
            object.__setattr__(self, "mu", mu)
        else:
            if mu is not None:
                raise ValueError(
                    "{} is not supported for mu".format(type(mu))
                )
            object.__setattr__(self, "mu", None)

    def set_precision(self, precision):
        if isinstance(precision, (int, float, np.number)):
            if self.shape == ():
                object.__setattr__(self, "precision", np.array(precision))
            else:
                object.__setattr__(self, "precision", np.ones(self.shape) * precision)
        elif isinstance(precision, np.ndarray):
            if precision.shape != self.shape:
                raise ValueError(
                    "The sizes of the arrays mu and precision don't match: {}, {}"
                    .format(self.shape, precision.shape)
                )
            object.__setattr__(self, "precision", precision)
        else:
            if precision is not None:
                raise ValueError(
                    "{} is not allowed for precision"
                    .format(type(precision))
                )
            object.__setattr__(self, "precision", precision)

    def __repr__(self):
        return (
            "Student's T"
            "(\nmu=\n{0.mu},\nprecision=\n{0.precision},\ndof={0.dof}\n)"
            .format(self)
        )

    @property
    def ndim(self):
        return self.mu.ndim

    @property
    def size(self):
        return self.mu.size

    @property
    def shape(self):
        return self.mu.shape

    @property
    def mean(self):
        if self.dof > 1:
            return self.mu
        else:
            raise AttributeError

    @property
    def var(self):
        if self.dof > 2:
            return self.dof / (self.dof - 2) / self.precision
        else:
            raise AttributeError

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.precision = 1 / np.var(X, axis=0)
        self.dof = 1
        params = np.hstack(
            (self.mu.ravel(),
             self.precision.ravel(),
             self.dof)
        )
        while True:
            E_eta, E_lneta = self._expectation(X)
            self._maximization(X, E_eta, E_lneta)
            new_params = np.hstack(
                (self.mu.ravel(),
                 self.precision.ravel(),
                 self.dof)
            )
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        d = X - self.mu
        a = 0.5 * (self.dof + 1)
        b = 0.5 * (self.dof + self.precision * d ** 2)
        E_eta = a / b
        E_lneta = digamma(a) - np.log(b)
        return E_eta, E_lneta

    def _maximization(self, X, E_eta, E_lneta):
        self.mu = np.sum(E_eta * X, axis=0) / np.sum(E_eta, axis=0)
        d = X - self.mu
        self.precision = 1 / np.mean(E_eta * d ** 2, axis=0)
        N = len(X)
        self.dof += 0.01 * (
            N * np.log(0.5 * self.dof) + N
            - N * digamma(0.5 * self.dof)
            + np.sum(E_lneta - E_eta, axis=0)
        )

    def _pdf(self, X):
        d = X - self.mu
        D_sq = self.precision * d ** 2
        return (
            gamma(0.5 * (self.dof + 1))
            * self.precision ** 0.5
            * (1 + D_sq / self.dof) ** (-0.5 * (1 + self.dof))
            / gamma(self.dof * 0.5)
            / (np.pi * self.dof) ** 0.5
        )

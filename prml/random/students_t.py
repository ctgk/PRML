import numpy as np
from scipy.special import gamma, digamma
from prml.random.random import RandomVariable
from prml.random.gamma import Gamma
from prml.random.gaussian import Gaussian


class StudentsT(RandomVariable):
    """
    Student's t-distribution
    p(x|mu, L(precision), dof)
    = (1 + (x-mu)^T @ L @ (x - mu) / dof)^-(D + dof)/2 / const.
    """

    def __init__(self, mu=None, precision=None, dof=None):
        assert mu is None or isinstance(mu, (int, float, np.ndarray))
        assert precision is None or isinstance(precision, (int, float, np.ndarray))
        assert dof is None or isinstance(dof, (int, float))
        self.mu = mu
        self.precision = precision
        self.dof = dof

    def __setattr__(self, name, value):
        if name is "mu":
            if isinstance(value, (int, float)):
                object.__setattr__(self, "ndim", 1)
                object.__setattr__(self, name, np.array([value]))
            elif isinstance(value, np.ndarray):
                assert value.ndim == 1
                object.__setattr__(self, "ndim", value.size)
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        elif name is "precision":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, np.eye(self.ndim) * value)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, "ndim", value.size)
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        else:
            object.__setattr__(self, name, value)


    def __repr__(self):
        return "Student's T(\nmu={0},\nprecision=\n{1},\ndof={2}\n)".format(self.mu, self.precision, self.dof)

    @property
    def mean(self):
        if self.dof > 1:
            return self.mu
        else:
            raise AttributeError

    @property
    def var(self):
        if self.dof > 2:
            return np.linalg.inv(self.precision) * self.dof / (self.dof - 2)
        else:
            raise AttributeError

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.precision = np.linalg.inv(np.atleast_2d(np.cov(X, rowvar=False)))
        self.dof = 1
        while True:
            params = np.hstack([self.mu.ravel(), self.precision.ravel(), self.dof])
            E_eta, E_lneta = self._expectation(X)
            self._maximization(X, E_eta, E_lneta)
            if np.allclose(params, np.hstack([self.mu.ravel(), self.precision.ravel(), self.dof])):
                break

    def _expectation(self, X):
        d = X - self.mu
        a = 0.5 * (self.dof + self.ndim)
        b = 0.5 * (self.dof + np.sum(d @ self.precision * d, -1))
        E_eta = a / b
        E_lneta = digamma(a) - np.log(b)
        return E_eta, E_lneta

    def _maximization(self, X, E_eta, E_lneta):
        self.mu = np.sum(E_eta[:, None] * X) / np.sum(E_eta)
        d = X - self.mu
        self.precision = np.linalg.inv(
            np.atleast_2d(np.cov(E_eta[:, None] ** 0.5 * d, rowvar=False, bias=True))
        )
        N = len(X)
        self.dof += 0.01 * (
            N * np.log(0.5 * self.dof) + N
            - N * digamma(0.5 * self.dof)
            + np.sum(E_lneta - E_eta)
        )

    def _proba(self, X):
        d = X - self.mu
        D_sq = np.sum(d @ self.precision * d, -1)
        return gamma(0.5 * (self.dof + 1)) * np.linalg.det(self.precision) ** 0.5 * (1 + D_sq / self.dof) ** (-0.5 * (self.ndim + self.dof)) / gamma(self.dof * 0.5) / np.power(np.pi * self.dof, 0.5 * self.ndim)

import numpy as np
from scipy.special import gamma, digamma


class StudentsTDistribution(object):
    """
    student's t-distribution

    p(x|mean,precision,dof)
    """

    def __init__(self, mean=0., precision=1., dof=1.):
        """
        construct student's t-distribution

        Parameters
        ----------
        mean : float
            initial mean
        precision : float
            initial precision
        dof : float
            initial degree of freedom
        """
        self.mean = mean
        self.precision = precision
        self.dof = dof

    def fit(self, x, learning_rate=0.01):
        """
        maximum likelihood estimation of student's t-distribution's parameters

        Parameters
        ----------
        x : (sample_size,) ndarray
            input
        learning_rate : float
            update ratio of a parameter
        """
        while True:
            params = [self.mean, self.precision, self.dof]
            E_eta, E_lneta = self._expectation(x)
            self._maximization(x, E_eta, E_lneta, learning_rate)
            if np.allclose(params, [self.mean, self.precision, self.dof]):
                break

    def _expectation(self, x):
        E_eta = (self.dof + 1) / (self.dof + self.precision * (x - self.mean) ** 2)
        E_lneta = digamma(0.5 * (self.dof + 1)) - np.log(0.5 * (self.dof + self.precision * (x - self.mean) ** 2))
        return E_eta, E_lneta

    def _maximization(self, x, E_eta, E_lneta, learning_rate):
        self.mean = np.sum(E_eta * x) / np.sum(E_eta)
        self.precision = 1 / np.mean(E_eta * (x - self.mean) ** 2)
        N = len(x)
        self.dof += learning_rate * (
            N * np.log(0.5 * self.dof) + N
            - N * digamma(0.5 * self.dof)
            + np.sum(E_lneta - E_eta))

    def proba(self, x):
        """
        compute probability density function

        Parameters
        ----------
        x : (sample_size,) ndarray
            input

        Returns
        -------
        output : (sample_size,) ndarray
            probabilitity
        """
        return ((1 + self.precision * (x - self.mean) ** 2 / self.dof) ** (-0.5 * (self.dof + 1))
                * gamma(0.5 * (self.dof + 1))
                * np.sqrt(self.precision / (np.pi * self.dof))
                / gamma(0.5 * self.dof))

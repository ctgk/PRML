import numpy as np
from scipy.special import gamma, digamma


class StudentsTDistribution(object):

    def fit(self, x, learning_rate=0.01):
        """
        maximum likelihood estimation of student's t-distribution's parameters

        Parameters
        ----------
        x : (sample_size,) ndarray
            input
        """
        self.mean = 0
        self.a = 1
        self.b = 1
        while True:
            params = [self.mean, self.a, self.b]
            self._expectation(x)
            self._maximization(x, learning_rate)
            if np.allclose(params, [self.mean, self.a, self.b]):
                break

    def _expectation(self, x):
        self.precisions = (self.a + 0.5) / (self.b + 0.5 * (x - self.mean) ** 2)

    def _maximization(self, x, learning_rate):
        self.mean = np.sum(self.precisions * x) / np.sum(self.precisions)
        a = self.a
        b = self.b
        self.a = a + learning_rate * (
            len(x) * np.log(b)
            + np.log(np.prod(self.precisions))
            - len(x) * digamma(a))
        self.b = a * len(x) / np.sum(self.precisions)

    def probability(self, x):
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
        return ((1 + (x - self.mean) ** 2/(2 * self.b)) ** (-self.a - 0.5)
                * gamma(self.a + 0.5)
                / (gamma(self.a) * np.sqrt(2 * np.pi * self.b)))

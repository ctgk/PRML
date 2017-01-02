import numpy as np
import scipy.special as sp


class Beta(object):

    def __init__(self, pseudo_ones, pseudo_zeros):
        self.pseudo_ones = pseudo_ones
        self.pseudo_zeros = pseudo_zeros
        self.n_ones = pseudo_ones
        self.n_zeros = pseudo_zeros

    def fit(self, n_ones=None, n_zeros=None):
        self.n_ones += 0 if n_ones is None else n_ones
        self.n_zeros += 0 if n_zeros is None else n_zeros

    def predict_proba(self, x):
        return sp.gamma(self.n_ones + self.n_zeros) * np.power(x, self.n_ones - 1) * np.power(1 - x, self.n_zeros - 1) / sp.gamma(self.n_ones) / sp.gamma(self.n_zeros)


class Gaussian(object):

    def fit(self, x):
        self.mean = np.mean(x)
        self.var = np.var(x)

    def predict_proba(self, x):
        return (np.exp(-0.5 * (x - self.mean) ** 2 / self.var)
                / np.sqrt(2 * np.pi * self.var))


class GaussianMixture(object):

    def __init__(self, n_component):
        self.n_component = n_component

    def fit(self, X, iter_max=10):
        self.ndim = np.size(X, 1)
        self.weights = np.ones(self.n_component) / self.n_component
        self.means = np.random.uniform(X.min(), X.max(), (self.ndim, self.n_component))
        self.covs = np.repeat(10 * np.eye(self.ndim), self.n_component).reshape(self.ndim, self.ndim, self.n_component)
        for i in range(iter_max):
            params = np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))
            resps = self._expectation(X)
            self._maximization(X, resps)
            if np.allclose(params, np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))):
                break
        else:
            print("parameters may not have converged")

    def _gauss(self, X):
        precisions = np.linalg.inv(self.covs.T).T
        diffs = X[:, :, None] - self.means
        exponents = np.sum(np.einsum('nik,ijk->njk', diffs, precisions) * diffs, axis=1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs.T).T * (2 * np.pi) ** self.ndim)

    def _expectation(self, X):
        resps = self.weights * self._gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.weights = Nk / len(X)
        self.means = X.T @ resps / Nk
        diffs = X[:, :, None] - self.means
        self.covs = np.einsum('nik,njk->ijk', diffs, diffs * np.expand_dims(resps, 1)) / Nk

    def predict_proba(self, X):
        gauss = self.weights * self._gauss(X)
        return np.sum(gauss, axis=-1)

    def classify(self, X):
        joint_prob = self.weights * self.gauss(X)
        return np.argmax(joint_prob, axis=1)


class StudentsT(object):

    def __init__(self, mean=0, a=1, b=1, learning_rate=0.01):
        self.mean = mean
        self.a = a
        self.b = b
        self.learning_rate = learning_rate

    def fit(self, x):
        while True:
            params = [self.mean, self.a, self.b]
            self._expectation(x)
            self._maximization(x)
            if np.allclose(params, [self.mean, self.a, self.b]):
                break

    def _expectation(self, x):
        self.precisions = (self.a + 0.5) / (self.b + 0.5 * (x - self.mean) ** 2)

    def _maximization(self, x):
        self.mean = np.sum(self.precisions * x) / np.sum(self.precisions)
        a = self.a
        b = self.b
        self.a = a + self.learning_rate * (
            len(x) * np.log(b)
            + np.log(np.prod(self.precisions))
            - len(x) * sp.digamma(a))
        self.b = a * len(x) / np.sum(self.precisions)

    def predict_proba(self, x):
        return ((1 + (x - self.mean) ** 2/(2 * self.b)) ** (-self.a - 0.5)
                * sp.gamma(self.a + 0.5)
                / (sp.gamma(self.a) * np.sqrt(2 * np.pi * self.b)))

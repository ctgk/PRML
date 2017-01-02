import functools
import itertools
import numpy as np


class Polynomial(object):

    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, x):
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.array(features).transpose()


class Gaussian(object):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def _gauss(self, x, mean, var):
        return np.exp(-0.5 * np.square(x - mean) / var)

    def transform(self, x):
        basis = [np.ones_like(x)]
        for m in self.mean:
            basis.append(self._gauss(x, m, self.var))
        return np.asarray(basis).transpose()


class Sigmoidal(object):

    def __init__(self, mean, coef=1):
        self.mean = mean
        self.coef = coef

    def _sigmoid(self, x, mean):
        return 1 / (1 + np.exp(self.coef * (mean - x)))

    def transform(self, x):
        basis = [np.ones_like(x)]
        for m in self.mean:
            basis.append(self._sigmoid(x, m))
        return np.asarray(basis).transpose()

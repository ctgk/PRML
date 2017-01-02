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

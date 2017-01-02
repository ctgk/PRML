import numpy as np
import scipy.special as sp


class Beta(object):

    def __init__(self, pseudo_ones, pseudo_zeros):
        self.pseudo_ones = pseudo_ones
        self.pseudo_zeros = pseudo_zeros
        self.n_ones = pseudo_ones
        self.n_zeros = pseudo_zeros

    def fit(self, n_ones, n_zeros):
        self.n_ones += 0 if n_ones is None else n_ones
        self.n_zeros += 0 if n_zeros is None else n_zeros

    def predict_dist(self, x):
        return sp.gamma(self.n_ones + self.n_zeros) * np.power(x, self.n_ones - 1) * np.power(1 - x, self.n_zeros - 1) / sp.gamma(self.n_ones) / sp.gamma(self.n_zeros)

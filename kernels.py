import numpy as np


class RBF(object):

    def __init__(self, params):
        self.params = params
        self.ndim = len(params) - 1

    def __call__(self, x, y):
        """
        calculate radial basis function
        k(x, y) = a * exp(-0.5 * c1 * (x1 - y1) ** 2 ...)

        Parameters
        ----------
        x : ndarray [..., ndim]
            input of this kernel function
        y : ndarray [..., ndim]
            another input

        Returns
        -------
        output : ndarray
            output of the kernel function
        """
        d = self.params[1:] * (x - y) ** 2
        return self.params[0] * np.exp(-0.5 * np.sum(d, axis=-1))

    def derivatives(self, x, y):
        d = np.sum((x - y) ** 2, axis=-1)
        delta = np.exp(-0.5 * self.params[1:] * d)
        deltas = -0.5 * d * delta * self.params[0]
        return np.array([delta, deltas])

    def update_parameters(self, updates):
        self.params += updates

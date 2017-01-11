import numpy as np


class RBF(object):

    def __init__(self, params):
        assert params.ndim == 1
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
        d = self.params[1:] * (x - y) ** 2
        delta = np.exp(-0.5 * np.sum(d, axis=-1))
        deltas = -0.5 * (x - y) ** 2 * (delta * self.params[0])[:, :, None]
        return np.concatenate((np.expand_dims(delta, 0), deltas.T))

    def update_parameters(self, updates):
        self.params += updates

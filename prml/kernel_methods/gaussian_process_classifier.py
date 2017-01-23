import numpy as np


class GaussianProcessClassifier(object):

    def __init__(self, kernel, nu=1e-4):
        """
        construct gaussian process classifier

        Parameters
        ----------
        kernel
            kernel function to be used to compute Gram matrix
        nu : float
            parameter to ensure the matrix to be positive
        """
        self.kernel = kernel
        self.nu = nu

    def _pairwise(self, x, y):
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1)))

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t):
        if X.ndim == 1:
            X = X[:, None]
        self.X = X
        self.t = t
        Gram = self.kernel(*self._pairwise(X, X))
        self.covariance = Gram + np.eye(len(Gram)) * self.nu
        self.precision = np.linalg.inv(self.covariance)

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(*self._pairwise(X, self.X))
        a_mean = K @ self.precision @ self.t
        return self._sigmoid(a_mean)

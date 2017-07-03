import numpy as np
from prml.linear.classifier import Classifier


class LeastSquaresClassifier(Classifier):
    """
    Least squares classifier model
    y = argmax_k X @ W
    """

    def __init__(self, W=None):
        self.W = W

    def __repr__(self):
        return "Gaussian(t|mean=X@{0.W})".format(self)

    def _ml(self, X, t):
        T = np.eye(int(np.max(t)) + 1)[t]
        self.W = np.linalg.pinv(X) @ T

    def _classify(self, X):
        return np.argmax(X @ self.W, axis=-1)

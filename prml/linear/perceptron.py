import numpy as np
from prml.linear.classifier import Classifier


class Perceptron(Classifier):
    """
    Perceptron model
    """

    def __init__(self, w=None):
        self.w = w

    def __repr__(self):
        return "sign(W@{})".format(self.w)

    def fit(self, X, t, max_epoch=100):
        """
        stochastic gradient descent method to estimate decision boundary

        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            input data
        t : (sample_size,) np.ndarray
            binary (-1, 1) target data
        max_epoch : int
            number of maximum epoch
        """
        self._check_input(X)
        self._check_target(t)
        self._check_binary_negative(t)
        self.w = np.zeros(np.size(X, 1))
        for _ in range(max_epoch):
            N = len(t)
            index = np.random.permutation(N)
            X = X[index]
            t = t[index]
            for x, label in zip(X, t):
                self.w += x * label
                if (X @ self.w * t > 0).all():
                    break
            else:
                continue
            break

    def _predict(self, X):
        return np.sign(X @ self.w).astype(np.int)


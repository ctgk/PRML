import numpy as np
from prml.linear.classifier import Classifier


class Perceptron(Classifier):
    """
    Perceptron model
    """

    def _fit(self, X, t, max_epoch=100):
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

    def _classify(self, X):
        return np.sign(X @ self.w).astype(np.int)

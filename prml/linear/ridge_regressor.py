import numpy as np
from prml.linear.regressor import Regressor


class RidgeRegressor(Regressor):
    """
    Ridge regression model
    w* = argmin |t - X @ w| + a * |w|_2^2
    """

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def _fit(self, X, t):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)

    def _predict(self, X):
        y = X @ self.w
        return y

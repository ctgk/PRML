import numpy as np
from prml.linear.regressor import Regressor


class RidgeRegressor(Regressor):
    """
    Ridge regression model
    w* = argmin |t - X @ w| + a * |w|_2^2
    """

    def __init__(self, precision=1.):
        self.precision = precision

    def _fit(self, X, t):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.precision * eye + X.T @ X, X.T @ t)

    def _predict(self, X):
        y = X @ self.w
        return y

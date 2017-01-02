import numpy as np


class LinearRegression(object):

    def fit(self, X, t):
        self.coef = np.linalg.pinv(X).dot(t)
        self.std = np.sqrt(np.mean((X.dot(self.coef) - t) ** 2))

    def predict(self, X):
        return X.dot(self.coef)

    def predict_dist(self, X):
        y = self.predict(X)
        return y, np.zeros_like(y) + self.std


class BayesianLinearRegression(object):

    def __init__(self, alpha=0.1, beta=0.25):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, t):
        self.w_var = np.linalg.inv(
            self.alpha * np.identity(np.size(X, 1))
            + self.beta * X.T.dot(X))
        self.w_mean = self.beta * self.w_var.dot(X.T.dot(t))

    def predict(self, X):
        return X.dot(self.w_mean)

    def predict_dist(self, X):
        y = self.predict(X)
        y_var = 1 / self.beta + np.sum(X.dot(self.w_var) * X, axis=1)
        y_std = np.sqrt(y_var)
        return y, y_std

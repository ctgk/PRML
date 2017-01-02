import numpy as np


class LogisticRegression(object):

    def __init__(self, alpha=0):
        self.alpha = alpha

    def _sigmoid(self, a):
        return np.divide(1, 1 + np.exp(-a))

    def fit(self, X, t, iter_max=100):
        self.w = np.zeros(np.size(X, 1))
        for i in range(iter_max):
            w = np.copy(self.w)
            y = self.predict_proba(X)
            grad = X.T @ (y - t) + self.alpha + w
            hessian = X.T @ np.diag(y * (1 - y)) @ X + self.alpha * np.eye(len(w))
            try:
                self.w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, self.w):
                break
        else:
            print("parameters may not have converged")

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.int)

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w)


class BayesianLogisticRegression(LogisticRegression):

    def fit(self, X, t, iter_max=100):
        super().fit(X, t, iter_max)
        y = self.predict_proba(X)
        hessian = X.T @ np.diag(y * (1 - y)) @ X + self.alpha * np.eye(len(w))
        self.w_cov = np.linalg.inv(hessian)

    def predict_dist(self, X):
        mu_a = X @ self.w
        var_a = np.sum(X @ self.w_cov * X, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))

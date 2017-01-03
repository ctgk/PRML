import numpy as np


class LinearRegression(object):

    def fit(self, X, t):
        self.coef = np.linalg.pinv(X).dot(t)
        self.var = np.mean(np.square(X @ self.coef - t))

    def predict(self, X):
        return X.dot(self.coef)


class RidgeRegression(LinearRegression):

    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def fit(self, X, t):
        self.coef = np.linalg.inv(
            self.alpha * np.eye(np.size(X, 1)) + X.T @ X) @ X.T @ t


class BayesianLinearRegression(object):

    def __init__(self, alpha=0.1, beta=0.25):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, t):
        self.w_var = np.linalg.inv(
            self.alpha * np.identity(np.size(X, 1))
            + self.beta * X.T @ X)
        self.w_mean = self.beta * self.w_var @ X.T @ t

    def fit_evidence(self, X, t, iter_max=100):
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        for i in range(iter_max):
            params = [self.alpha, self.beta]
            self.fit(X, t)
            self.gamma = np.sum(self.beta * eigenvalues / (self.alpha + self.beta * eigenvalues))
            self.alpha = self.gamma / (self.w_mean @ self.w_mean)
            self.beta = (len(t) - self.gamma) / np.sum(np.square(t - X @ self.w_mean))
            if np.allclose(params, [self.alpha, self.beta]):
                break
        else:
            print("parameters may not have converged")

    def log_evidence(self, X, t):
        M = X.T @ X
        return (
            len(M) * np.log(self.alpha)
            + len(t) * np.log(self.beta)
            - self.beta * np.sum(np.square(t - X @ self.w_mean))
            - np.linalg.slogdet(self.alpha + self.beta * M)[1]
        )

    def predict(self, X):
        return X.dot(self.w_mean)

    def predict_dist(self, X):
        y = self.predict(X)
        y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis=1)
        y_std = np.sqrt(y_var)
        return y, y_std


class GaussianProcessRegression(object):

    def __init__(self, kernel, beta=1.):
        self.kernel = kernel
        self.beta = beta

    def _pairwise(self, x, y):
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1))
        )

    def fit(self, X, t):
        self.X = X
        self.t = t
        Gram = self.kernel(*self._pairwise(X, X))
        self.covariance = Gram + np.identity(len(X)) / self.beta
        self.precision = np.linalg.inv(self.covariance)

    def fit_kernel(self, X, t, learning_rate=0.1, iter_max=10000):
        for i in range(iter_max):
            params = np.copy(self.kernel.params)
            self.fit(X, t)
            gradients = self.kernel.derivatives(*self._pairwise(X, X))
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
            self.kernel.update_parameters(learning_rate * updates)
            if np.allclose(params, self.kernel.params):
                break
        else:
            print("parameters may not have converged")

    def predict(self, X):
        K = self.kernel(*self._pairwise(X, self.X))
        mean = K @ self.precision @ self.t
        return mean

    def predict_dist(self, X):
        K = self.kernel(*self._pairwise(X, self.X))
        mean = K @ self.precision @ self.t
        var = self.kernel(X, X) + 1 / self.beta - np.sum(K @ self.precision * K, axis=1)
        return mean.ravel(), np.sqrt(var.ravel())

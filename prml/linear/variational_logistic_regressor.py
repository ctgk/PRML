import numpy as np
from prml.linear.logistic_regressor import LogisticRegressor


class VariationalLogisticRegressor(LogisticRegressor):

    def __init__(self, alpha=None, a0=1., b0=1.):
        """
        construct variational logistic regressor
        Parameters
        ----------
        alpha : float or None
            precision parameter of the prior
            if None, this is also the subject to estimate
        a0 : float
            a parameter of hyper prior Gamma dist.
            Gamma(alpha|a0,b0)
            if alpha is not None, this argument will be ignored
        b0 : float
            another parameter of hyper prior Gamma dist.
            Gamma(alpha|a0,b0)
            if alpha is not None, this argument will be ignored
        """
        if alpha is not None:
            self.__alpha = alpha
        else:
            self.a0 = a0
            self.b0 = b0

    def _fit(self, X, t, iter_max=1000):
        assert X.ndim == 2
        assert t.ndim == 1
        N, M = X.shape
        if hasattr(self, "a0"):
            self.a = self.a0 + 0.5 * M
        xi = np.random.uniform(-1, 1, size=N)
        assert xi.shape == (N,), xi.shape
        I = np.eye(M)
        param = np.copy(xi)
        for i in range(iter_max):
            lambda_ = np.tanh(xi) * 0.25 / xi
            assert lambda_.shape == (N,), lambda_.shape
            self.w_var = np.linalg.inv(
                I / self.alpha
                + 2 * (lambda_ * X.T) @ X)
            self.w_mean = self.w_var @ np.sum(X.T * (t - 0.5), axis=1)
            xi = np.sqrt(np.sum(X @ (self.w_var + self.w_mean * self.w_mean[:, None]) * X, axis=-1))
            if np.allclose(xi, param):
                break
            else:
                param = np.copy(xi)
        self.n_iter = i + 1

    @property
    def alpha(self):
        if hasattr(self, "__alpha"):
            return self.__alpha
        else:
            try:
                self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            except AttributeError:
                self.b = self.b0
            return self.a / self.b

    def _proba(self, X):
        mu_a = X @ self.w_mean
        var_a = np.sum(X @ self.w_var * X, axis=1)
        y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y

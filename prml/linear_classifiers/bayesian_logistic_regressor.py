import numpy as np
from .logistic_regressor import LogisticRegressor


class BayesianLogisticRegressor(LogisticRegressor):

    def fit(self, X, t, iter_max=100):
        """
        baysian estimation of weight parameter with laplace approximation

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels
        iter_max : int
            number of maximum iterations

        Attributes
        ----------
        w_mean : (n_features,) ndarray
            mean of laplace approximation of the posterior
        w_var : (n_features, n_features) ndarray
            variance of laplace approximated posterior
        n_iter : int
            number of iterations took until convergence
        """
        assert X.ndim == 2
        assert t.ndim == 1
        self.w_mean = np.zeros(np.size(X, 1))
        I = np.eye(len(self.w_mean))
        for i in range(iter_max):
            w = np.copy(self.w_mean)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t) + self.alpha * w
            hessian = (X.T * y * (1 - y)) @ X + self.alpha * I
            try:
                self.w_mean -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, self.w_mean):
                break
        self.n_iter = i + 1
        self.w_var = np.linalg.inv(hessian)

    def predict_proba(self, X):
        """
        predictive distribution of input belonging class one

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input

        Returns
        -------
        output : ndarray (sample_size,)
            probability of class one for each input
        """
        assert X.ndim == 2
        mu_a = X @ self.w_mean
        var_a = np.sum(X @ self.w_var * X, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))

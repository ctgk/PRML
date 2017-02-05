import numpy as np


class VariationalLinearRegressor(object):

    def __init__(self, beta=1., a0=1., b0=1.):
        """
        construct variational linear regressor

        Parameters
        ----------
        beta : float
            precision of observation noise
        a0 : float
            a parameter of prior gamma distribution
            Gamma(alpha|a0,b0)
        b0 : float
            another parameter of prior gamma distribution
            Gamma(alpha|a0,b0)
        """
        self.beta = beta
        self.a0 = a0
        self.b0 = b0

    def fit(self, X, t, iter_max=100):
        """
        variational bayesian estimation the parameters
        p(w,alpha|X,t)
        ~ q(w)q(alpha)
        = N(w|w_mean, w_var)Gamma(alpha|a,b)

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        t : (sample_size,) ndarray
            corresponding target data
        iter_max : int
            maximum number of iterations

        Returns
        -------
        a : float
            a parameter of variational posterior gamma distribution
        b : float
            another parameter of variational posterior gamma distribution
        w_mean : (n_features,) ndarray
            mean of variational posterior gaussian distribution
        w_var : (n_features, n_feautures) ndarray
            variance of variational posterior gaussian distribution
        n_iter : int
            number of iterations performed
        """
        assert X.ndim == 2
        assert t.ndim == 1
        self.a = self.a0 + 0.5 * np.size(X, 1)
        self.b = self.b0
        I = np.eye(np.size(X, 1))
        for i in range(iter_max):
            param = self.b
            self.w_var = np.linalg.inv(
                self.a * I / self.b
                + self.beta * X.T @ X)
            self.w_mean = self.beta * self.w_var @ X.T @ t
            self.b = self.b0 + 0.5 * (
                np.sum(self.w_mean ** 2)
                + np.trace(self.w_var))
            if np.allclose(self.b, param):
                break
        self.n_iter = i + 1

    def predict(self, X, with_error=True):
        """
        predict outputs of this model

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        with_error : bool
            returns standard deviation of the prediction if True

        Returns
        -------
        y : (sample_size,) ndarray
            mean of the predictive distribution
        y_std : (sample_size,) ndarray
            std of the predictive distribution
        """
        assert X.ndim == 2
        y = X @ self.w_mean
        if with_error:
            y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

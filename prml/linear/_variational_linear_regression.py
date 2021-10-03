import numpy as np

from prml.linear._regression import Regression


class VariationalLinearRegression(Regression):
    """Variational bayesian linear regression model.

    p(w,alpha|X,t)
    ~ q(w)q(alpha)
    = N(w|w_mean, w_var)Gamma(alpha|a,b)

    Attributes
    ----------
    a : float
        a parameter of variational posterior gamma distribution
    b : float
        another parameter of variational posterior gamma distribution
    w_mean : (n_features,) ndarray
        mean of variational posterior gaussian distribution
    w_var : (n_features, n_features) ndarray
        variance of variational posterior gaussian distribution
    n_iter : int
        number of iterations performed
    """

    def __init__(self, beta: float = 1., a0: float = 1., b0: float = 1.):
        """Initialize variational linear regression model.

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

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        iter_max: int = 100,
    ):
        """Variational bayesian estimation of parameter.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        y_train : np.ndarray
            training dependent variable (N,)
        iter_max : int, optional
            maximum number of iteration (the default is 100)
        """
        xtx = x_train.T @ x_train
        d = np.size(x_train, 1)
        self.a = self.a0 + 0.5 * d
        self.b = self.b0
        eye = np.eye(d)
        for _ in range(iter_max):
            param = self.b
            self.w_var = np.linalg.inv(self.a * eye / self.b + self.beta * xtx)
            self.w_mean = self.beta * self.w_var @ x_train.T @ y_train
            self.b = (
                self.b0
                + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            )
            if np.allclose(self.b, param):
                break

    def predict(self, x: np.ndarray, return_std: bool = False):
        """Return predictions.

        Parameters
        ----------
        x : np.ndarray
            Input independent variable (N, D)
        return_std : bool, optional
            return standard deviation of predictive distribution if True
            (the default is False)

        Returns
        -------
        y :  np.ndarray
            mean of predictive distribution (N,)
        y_std : np.ndarray
            standard deviation of predictive distribution (N,)
        """
        y = x @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(x @ self.w_var * x, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

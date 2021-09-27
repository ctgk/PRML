import typing as tp

import numpy as np

from prml.linear._logistic_regression import LogisticRegression


class VariationalLogisticRegression(LogisticRegression):
    """Variational logistic regression model.

    Graphical Model
    ---------------

    ```txt
    *----------------*
    |                |               ****  alpha
    |     phi_n      |             **    **
    |       **       |            *        *
    |       **       |            *        *
    |       |        |             **    **
    |       |        |               ****
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       v        |                v
    |      ****      |               ****  w
    |    **    **    |             **    **
    |   *        *   |            *        *
    |   *        *<--|------------*        *
    |    **    **    |             **    **
    |  t_n ****      |               ****
    |             N  |
    *----------------*
    ```
    """

    def __init__(
        self,
        alpha: tp.Optional[float] = None,
        a0: float = 1.,
        b0: float = 1.,
    ):
        """Construct variational logistic regression model.

        Parameters
        ----------
        alpha : tp.Optional[float]
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

    def fit(self, x_train: np.ndarray, t: np.ndarray, iter_max: int = 1000):
        """Variational bayesian estimation of the parameter.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        t : np.ndarray
            training dependent variable (N,)
        iter_max : int, optional
            maximum number of iteration (the default is 1000)
        """
        n, d = x_train.shape
        if hasattr(self, "a0"):
            self.a = self.a0 + 0.5 * d
        xi = np.random.uniform(-1, 1, size=n)
        eye = np.eye(d)
        param = np.copy(xi)
        for _ in range(iter_max):
            lambda_ = np.tanh(xi) * 0.25 / xi
            self.w_var = np.linalg.inv(
                eye / self.alpha + 2 * (lambda_ * x_train.T) @ x_train)
            self.w_mean = self.w_var @ np.sum(x_train.T * (t - 0.5), axis=1)
            xi = np.sqrt(np.sum(
                (
                    x_train
                    @ (self.w_var + self.w_mean * self.w_mean[:, None])
                    * x_train
                ),
                axis=-1,
            ))
            if np.allclose(xi, param):
                break
            else:
                param = np.copy(xi)

    @property
    def alpha(self) -> float:
        """Return expectation of variational distribution of alpha.

        Returns
        -------
        float
            Expectation of variational distribution of alpha.
        """
        if hasattr(self, "__alpha"):
            return self.__alpha
        else:
            try:
                self.b = self.b0 + 0.5 * (
                    np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            except AttributeError:
                self.b = self.b0
            return self.a / self.b

    def proba(self, x: np.ndarray):
        """Return probability of input belonging class 1.

        Parameters
        ----------
        x : np.ndarray
            Input independent variable (N, D)

        Returns
        -------
        np.ndarray
            probability of positive (N,)
        """
        mu_a = x @ self.w_mean
        var_a = np.sum(x @ self.w_var * x, axis=1)
        y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y

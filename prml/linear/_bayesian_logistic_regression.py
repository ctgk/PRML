import numpy as np

from prml.linear._logistic_regression import LogisticRegression


class BayesianLogisticRegression(LogisticRegression):
    """Bayesian logistic regression model.

    w ~ Gaussian(0, alpha^(-1)I)
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)

    Graphical Model
    ---------------

    ```txt
    *----------------*
    |                |
    |      x_n       |               alpha
    |       **       |                **
    |       **       |                **
    |       |        |                |
    |       |        |                |
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

    def __init__(self, alpha: float = 1.):
        """Initialize bayesian logistic regression model.

        Parameters
        ----------
        alpha : float, optional
            Precision parameter of the prior, by default 1.
        """
        self.alpha = alpha

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_iter: int = 100,
    ):
        """Bayesian estimation of the posterior using Laplace approximation.

        Parameters
        ----------
        x_train : np.ndarray
            Training data independent variable (N, D)
        y_train : np.ndarray
            training data dependent variable (N,)
            binary 0 or 1
        max_iter : int, optional
            maximum number of paramter update iteration (the default is 100)
        """
        w = np.zeros(np.size(x_train, 1))
        eye = np.eye(np.size(x_train, 1))
        self.w_mean = np.copy(w)
        self.w_precision = self.alpha * eye
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(x_train @ w)
            grad = (
                x_train.T @ (y - y_train)
                + self.w_precision @ (w - self.w_mean)
            )
            hessian = (x_train.T * y * (1 - y)) @ x_train + self.w_precision
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian

    def proba(self, x: np.ndarray):
        """Return probability of input belonging class 1.

        Parameters
        ----------
        x : np.ndarray
            training data independent variable (N, D)

        Returns
        -------
        np.ndarray
            probability of positive (N,)
        """
        mu_a = x @ self.w_mean
        var_a = np.sum(np.linalg.solve(self.w_precision, x.T).T * x, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))

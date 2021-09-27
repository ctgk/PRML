import numpy as np

from prml.linear._regression import Regression


class RidgeRegression(Regression):
    """Ridge regression model.

    w* = argmin |t - X @ w| + alpha * |w|_2^2
    """

    def __init__(self, alpha: float = 1.):
        """Initialize ridge linear regression model.

        Parameters
        ----------
        alpha : float, optional
            Coefficient of the prior term, by default 1.
        """
        self.alpha = alpha

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """Maximum a posteriori estimation of parameter.

        Parameters
        ----------
        x_train : np.ndarray
            training data independent variable (N, D)
        y_train : np.ndarray
            training data dependent variable (N,)
        """
        eye = np.eye(np.size(x_train, 1))
        self.w = np.linalg.solve(
            self.alpha * eye + x_train.T @ x_train,
            x_train.T @ y_train,
        )

    def predict(self, x: np.ndarray):
        """Return prediction.

        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)

        Returns
        -------
        np.ndarray
            prediction of each input (N,)
        """
        return x @ self.w

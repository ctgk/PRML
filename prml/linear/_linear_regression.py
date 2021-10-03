import numpy as np

from prml.linear._regression import Regression


class LinearRegression(Regression):
    """Linear regression model.

    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """Perform least squares fitting.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        y_train : np.ndarray
            training dependent variable (N,)
        """
        self.w = np.linalg.pinv(x_train) @ y_train
        self.var = np.mean(np.square(x_train @ self.w - y_train))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """Return prediction given input.

        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)
        return_std : bool, optional
            returns standard deviation of each predition if True

        Returns
        -------
        y : np.ndarray
            prediction of each sample (N,)
        y_std : np.ndarray
            standard deviation of each predition (N,)
        """
        y = x @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y

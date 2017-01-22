import numpy as np


class RidgeRegressor(object):

    def __init__(self, alpha=1e-3):
        """
        construct ridge regressor

        Parameters
        ----------
        alpha : float
            coefficient of l2 norm of weight
        """
        assert isinstance(alpha, float) or isinstance(alpha, int)
        self.alpha = alpha

    def fit(self, X, t):
        """
        maximum a posteriori estimation

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data

        Attributes
        ----------
        w : ndarray (n_features,)
            coefficient of each feature
        """
        assert X.ndim == 2
        assert t.ndim == 1
        self.w = np.linalg.inv(
            self.alpha * np.eye(np.size(X, 1)) + X.T @ X) @ X.T @ t

    def predict(self, X):
        """
        predict outputs with this model
        p(y| X @ w)

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            samples to predict their output distributions

        Returns
        -------
        y : ndarray (sample_size,)
            mean of Gaussian distribution
        """
        assert X.ndim == 2
        y = X @ self.w
        return y

import numpy as np


class LinearRegressor(object):
    """
    Linear regressor by least squares error
    """

    def fit(self, X, t):
        """
        maximum likelihood estimation

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input training data
        t : ndarray (sample_size,)
            target

        Attributes
        ----------
        w : ndarray (n_features,)
            weight parameter of each feature
        var : float
            variance
        aic : float
            Akaike information criterion
        """
        assert X.ndim == 2
        assert t.ndim == 1
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))
        self.aic = -0.5 * len(X) * (np.log(2 * np.pi * self.var) + 1) - len(self.w)

    def predict(self, X, with_error=False):
        """
        predict outputs with this model
        p(y| X @ w, var)

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            samples to predict their output distributions
        with_error : bool
            return standard deviation of prediction

        Returns
        -------
        y : ndarray (sample_size,)
            mean of Gaussian distribution
        y_std : ndarray (sample_size,)
            standard deviation of Gaussian distribution
        """
        assert X.ndim == 2
        y = X @ self.w
        if with_error:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y

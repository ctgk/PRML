import numpy as np
from prml.linear._regression import Regression


class MultivariateGaussian(object):

    def __init__(self, mean=None, covariance=None):
        self.mean = mean
        self.covariance = covariance


class A:
    r"""
    Bayesian linear regression model

    .. math::
        p(y|{\bf x}, \alpha, \beta)
            = \int \mathcal{N}(y|{\bf w}^{\rm T}{\bf x}, {1\over\beta})
            \cdot \mathcal{N}({\bf w}|0, {1\over\alpha}{\rm I}) {\rm d}{\bf w}

    Attributes
    ----------
    alpha : float
        precision parameter of prior distribution
    beta : float
        precision parameter of likelihood function
    w_mean : np.ndarray (D,), None
        mean of posterior distribution :math:`{\bf w}`
    w_covariance : np.ndarray (D, D), None
        covariance of posterior distribution :math:`{\bf w}`
    w_precision : np.ndarray (D, D), None
        precision of posterior distribution :math:`{\bf w}`
        which is the inverse of covariance
    """


class BayesianRegression(Regression):
    r"""
    Bayesian linear regression model

    .. math::
        p(y|{\bf x}, \alpha, \beta)
            = \int \mathcal{N}(y|{\bf w}^{\rm T}{\bf x}, {1\over\beta})
            \cdot \mathcal{N}({\bf w}|0, {1\over\alpha}{\rm I}) {\rm d}{\bf w}

    Attributes
    ----------
    alpha : float
        precision parameter of prior distribution
    beta : float
        precision parameter of likelihood function
    w_mean : np.ndarray (D,), None
        mean of posterior distribution :math:`{\bf w}`
    w_covariance : np.ndarray (D, D), None
        covariance of posterior distribution :math:`{\bf w}`
    w_precision : np.ndarray (D, D), None
        precision of posterior distribution :math:`{\bf w}`
        which is the inverse of covariance
    """

    def __init__(self, alpha: float = 1., beta: float = 1.):
        """
        Initialize bayesian linear regression model.

        Parameters
        ----------
        alpha : float, optional
            precision parameter of prior distribution, by default 1.
        beta : float, optional
            precision parameter of likelihood function, by default 1.
        """
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim: int) -> tuple:
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        bayesian update of parameter given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable
        t : (N,) np.ndarray
            dependent variable
        """

        mean_prev, precision_prev = self._get_prior(np.size(X, 1))

        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self,
                X: np.ndarray,
                return_std: bool = False,
                sample_size: int = None):
        """
        return mean (and standard deviation) of predictive distribution

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            independent variable
        return_std : bool, optional
            flag to return standard deviation (the default is False)
        sample_size : int, optional
            number of samples to draw from the predictive distribution
            (the default is None, no sampling from the distribution)

        Returns
        -------
        y : (N,) np.ndarray
            mean of the predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of the predictive distribution
        y_sample : (N, sample_size) np.ndarray
            samples from the predictive distribution
        """

        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = X @ w_sample.T
            return y_sample
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

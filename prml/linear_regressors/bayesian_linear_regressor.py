import numpy as np


class BayesianLinearRegressor(object):

    def __init__(self, alpha=0.1, beta=0.25, n_features=None):
        """
        set hyperparameters

        Parameters
        ----------
        alpha : float
            precision of prior distribution for w
        beta : float
            precision of likelihood
        """
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, t):
        """
        bayesian estimation of posterior distribution of weight parameter

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data

        Attributes
        ----------
        w_mean : ndarray (n_features,)
            mean of Gaussian posterior distribution of weight
        w_var : ndarray (n_features, n_features)
        """
        assert X.ndim == 2, X.ndim
        assert t.ndim == 1, t.ndim
        if not hasattr(self, "w_var"):
            self.w_var = np.eye(np.size(X, 1)) / self.alpha
        if not hasattr(self, "w_mean"):
            self.w_mean = np.zeros(np.size(X, 1))
        w_cov = np.linalg.inv(self.w_var)
        self.w_var = np.linalg.inv(
            w_cov + self.beta * X.T @ X)
        self.w_mean = self.w_var @ (w_cov @ self.w_mean + self.beta * X.T @ t)

    def maximize_evidence(self, X, t, iter_max=100):
        """
        maximize evidence function
        Evidence Approximation, Empirical Bayes, type 2 maximum likelihood

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        w_mean : ndarray (n_features,)
            mean of Gaussian posterior distribution of weight
        w_var : ndarray (n_features, n_features)
            variance of Gaussian posteriror distribution of weight
        n_iter : int
            number of iterations took until convergence
        """
        assert X.ndim == 2
        assert t.ndim == 1
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        for i in range(iter_max):
            params = [self.alpha, self.beta]
            self.fit(X, t)
            gamma = np.sum(self.beta * eigenvalues / (self.alpha + self.beta * eigenvalues))
            self.alpha = gamma / (self.w_mean @ self.w_mean).clip(min=1e-10)
            self.beta = (len(t) - gamma) / np.sum(np.square(t - X @ self.w_mean))
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.n_iter = i + 1

    def log_evidence(self, X, t):
        """
        log evidence function

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data

        Returns
        -------
        output : float
            log evidence
        """
        assert X.ndim == 2
        assert t.ndim == 1
        M = X.T @ X
        return 0.5 * (
            len(M) * np.log(self.alpha)
            + len(t) * np.log(self.beta)
            - self.beta * np.sum(np.square(t - X @ self.w_mean))
            - self.alpha * self.w_mean @ self.w_mean
            - np.linalg.slogdet(self.alpha * np.eye(len(M)) + self.beta * M)[1]
            - len(t) * np.log(2 * np.pi)
        )

    def predict(self, X, with_error=True):
        """
        predict outputs of this model

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        with_error : bool
            return standard deviation of prediction if True

        Returns
        -------
        y : ndarray (sample_size,)
            mean of predictive distribution
        y_std : ndarray (sample_size,)
            standard deviation of predictive distribution
        """
        assert X.ndim == 2
        y = X @ self.w_mean
        if with_error:
            y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

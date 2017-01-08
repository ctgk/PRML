import numpy as np


class LinearRegression(object):

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
        coef : ndarray (n_features,)
            coefficient of each feature
        var : float
            variance
        aic : float
            Akaike information criterion
        """
        self.coef = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.coef - t))
        self.aic = -0.5 * len(X) * (np.log(2 * np.pi * self.var) + 1) - len(self.coef)

    def predict(self, X):
        """
        prediction with this model

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            samples to predict their outputs

        Returns
        -------
        output : ndarray (sample_size,)
            predictions
        """
        return X @ self.coef

    def predict_dist(self, X):
        """
        distribution of predictions with this model

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            samples to predict their output distributions

        Returns
        -------
        y : ndarray (sample_size,)
            mean of Gaussian distribution
        y_std : ndarray (sample_size,)
            standard deviation of Gaussian distribution
        """
        y = X @ self.coef
        y_std = np.sqrt(self.var) + np.zeros_like(y)
        return y, y_std

    def nll(self, X, t):
        """
        negative log likelihood

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            output data

        Returns
        -------
        nll : float
            negative log likelihood of the parameters given the data
        """
        return (
            0.5 * len(X) * np.log(2 * np.pi * self.var)
            + 0.5 * np.sum(np.square(self.predict(X) - t)) / self.var
        )


class RidgeRegression(LinearRegression):

    def __init__(self, alpha=1e-3):
        """
        set penalty for regularization term

        Parameters
        ----------
        alpha : float
            penalty coefficient of regularization
        """
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

        Returns
        -------
        coef : ndarray (n_features,)
            coefficient of each feature
        """
        self.coef = np.linalg.inv(
            self.alpha * np.eye(np.size(X, 1)) + X.T @ X) @ X.T @ t


class BayesianLinearRegression(object):

    def __init__(self, alpha=0.1, beta=0.25, n_features=None):
        """
        set hyperparameters

        Parameters
        ----------
        alpha : float
            precision of prior distribution for w
        beta : float
            precision of likelihood function
        """
        self.alpha = alpha
        self.beta = beta

        if n_features is not None:
            self.w_mean = np.zeros(n_features)
            self.w_var = np.eye(n_features) / self.alpha

    def fit(self, X, t):
        """
        calculate posterior distribution of weight parameter

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
            variance of Gaussian posteriror distribution of weight
        """
        self.w_var = np.linalg.inv(
            self.alpha * np.identity(np.size(X, 1))
            + self.beta * X.T @ X)
        self.w_mean = self.beta * self.w_var @ X.T @ t

    def fit_online(self, X, t):
        """
        sequential estimation of posterior distribution of weight parameter

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
        w_cov = np.linalg.inv(self.w_var)
        self.w_var = np.linalg.inv(
            w_cov + self.beta * X.T @ X)
        self.w_mean = self.w_var @ (w_cov @ self.w_mean + self.beta * X.T @ t)

    def fit_evidence(self, X, t, iter_max=100):
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
        alpha : float
            estimated hyperparameter of weight prior distribution
        beta : float
            estimated precision parameter
        w_mean : ndarray (n_features,)
            mean of Gaussian posterior distribution of weight
        w_var : ndarray (n_features, n_features)
            variance of Gaussian posteriror distribution of weight
        """
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
        else:
            print("parameters may not have converged")

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
        M = X.T @ X
        return 0.5 * (
            len(M) * np.log(self.alpha)
            + len(t) * np.log(self.beta)
            - self.beta * np.sum(np.square(t - X @ self.w_mean))
            - self.alpha * self.w_mean @ self.w_mean
            - np.linalg.slogdet(self.alpha * np.eye(len(M)) + self.beta * M)[1]
            - len(t) * np.log(2 * np.pi)
        )

    def predict(self, X):
        """
        mean of predictive distributions

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        y : ndarray (sample_size)
            mean of predictive distribution
        """
        return X.dot(self.w_mean)

    def predict_dist(self, X):
        """
        predictive distributions

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        y : ndarray (sample_size,)
            mean of predictive distribution
        y_std : ndarray (sample_size,)
            standard deviation of predictive distribution
        """
        y = self.predict(X)
        y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis=1)
        y_std = np.sqrt(y_var)
        return y, y_std


class GaussianProcessRegression(object):

    def __init__(self, kernel, beta=1.):
        self.kernel = kernel
        self.beta = beta

    def _pairwise(self, x, y):
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1))
        )

    def fit(self, X, t):
        self.X = X
        self.t = t
        Gram = self.kernel(*self._pairwise(X, X))
        self.covariance = Gram + np.identity(len(X)) / self.beta
        self.precision = np.linalg.inv(self.covariance)

    def fit_kernel(self, X, t, learning_rate=0.1, iter_max=10000):
        for i in range(iter_max):
            params = np.copy(self.kernel.params)
            self.fit(X, t)
            gradients = self.kernel.derivatives(*self._pairwise(X, X))
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
            self.kernel.update_parameters(learning_rate * updates)
            if np.allclose(params, self.kernel.params):
                break
        else:
            print("parameters may not have converged")

    def predict(self, X):
        K = self.kernel(*self._pairwise(X, self.X))
        mean = K @ self.precision @ self.t
        return mean

    def predict_dist(self, X):
        K = self.kernel(*self._pairwise(X, self.X))
        mean = K @ self.precision @ self.t
        var = self.kernel(X, X) + 1 / self.beta - np.sum(K @ self.precision * K, axis=1)
        return mean.ravel(), np.sqrt(var.ravel())

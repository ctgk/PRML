import numpy as np


class GaussianProcessRegressor(object):

    def __init__(self, kernel, beta=1.):
        """
        construct gaussian process regressor

        Parameters
        ----------
        kernel
            kernel function
        beta : float
            precision parameter of observation noise
        """
        self.kernel = kernel
        self.beta = beta

    def _pairwise(self, x, y):
        return (
            np.tile(x, (len(y), 1, 1)).transpose(1, 0, 2),
            np.tile(y, (len(x), 1, 1))
        )

    def fit(self, X, t):
        """
        estimate gaussian process

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input
        t : ndarray (sample_size,)
            corresponding target

        Attributes
        ----------
        covariance : ndarray (sample_size, sample_size)
            variance covariance matrix of gaussian process
        precision : ndarray (sample_size, sample_size)
            precision matrix of gaussian process
        """
        if X.ndim == 1:
            X = X[:, None]
        self.X = X
        self.t = t
        Gram = self.kernel(*self._pairwise(X, X))
        self.covariance = Gram + np.identity(len(X)) / self.beta
        self.precision = np.linalg.inv(self.covariance)

    def fit_kernel(self, X, t, learning_rate=0.1, iter_max=100):
        """
        maximum likelihood estimation of parameters in kernel function

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input
        t : ndarray (sample_size,)
            corresponding target
        learning_rate : float
            updation coefficient
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        covariance : ndarray (sample_size, sample_size)
            variance covariance matrix of gaussian process
        precision : ndarray (sample_size, sample_size)
            precision matrix of gaussian process

        Returns
        -------
        log_likelihood_list : list
            list of log likelihood value at each iteration
        """
        if X.ndim == 1:
            X = X[:, None]
        log_likelihood_list = [-np.Inf]
        self.fit(X, t)
        for i in range(iter_max):
            gradients = self.kernel.derivatives(*self._pairwise(X, X))
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
            for j in range(iter_max):
                self.kernel.update_parameters(learning_rate * updates)
                self.fit(X, t)
                log_like = self.log_likelihood()
                if log_like > log_likelihood_list[-1]:
                    log_likelihood_list.append(log_like)
                    break
                else:
                    self.kernel.update_parameters(-learning_rate * updates)
                    learning_rate *= 0.9
        log_likelihood_list.pop(0)
        return log_likelihood_list

    def log_likelihood(self):
        return -0.5 * (
            np.linalg.slogdet(self.covariance)[1]
            + self.t @ self.precision @ self.t
            + len(self.t) * np.log(2 * np.pi))

    def predict(self, X):
        """
        mean of the gaussian process

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input

        Returns
        -------
        mean : ndarray (sample_size,)
            predictions of corresponding inputs
        """
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(*self._pairwise(X, self.X))
        mean = K @ self.precision @ self.t
        return mean

    def predict_dist(self, X):
        """
        mean and std. of the gaussian process

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input

        Returns
        -------
        mean : ndarray (sample_size,)
            mean of the gaussian process given input
        std : ndarray (sample_size,)
            standard derivation of the gaussian process given input
        """
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(*self._pairwise(X, self.X))
        mean = K @ self.precision @ self.t
        var = self.kernel(X, X) + 1 / self.beta - np.sum(K @ self.precision * K, axis=1)
        return mean.ravel(), np.sqrt(var.ravel())

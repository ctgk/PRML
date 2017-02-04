import numpy as np
from scipy.misc import logsumexp


class BernoulliMixtureDistribution(object):

    def __init__(self, n_components):
        """
        construct mixture of bernoulli model

        Parameters
        ----------
        n_components : int
            numpber of bernoulli components
        """
        self.n_components = n_components

    def fit(self, X, iter_max=100):
        """
        maximum likelihood estimation of parameters via EM alg.

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input binary data, all elements are either 0 or 1
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        ndim : int
            dimensionality of data space
        weights : (n_components,) ndarray
            mixing coefficient of each bernoulli component
        means : (n_components, ndim) ndarray
            mean of each bernoulli component
        n_iter : int
            number of iterations performed
        """
        self.ndim = np.size(X, 1)
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = np.random.uniform(0.25, 0.75, size=(self.n_components, self.ndim))
        self.means /= np.sum(self.means, axis=-1, keepdims=True)
        for i in range(iter_max):
            params = np.hstack((self.weights.ravel(), self.means.ravel()))
            stats = self._expectation(X)
            self._maximization(X, stats)
            if np.allclose(params, np.hstack((self.weights.ravel(), self.means.ravel()))):
                break
        self.n_iter = i + 1

    def _log_bernoulli(self, X):
        np.clip(self.means, 1e-10, 1 - 1e-10, out=self.means)
        return np.sum(X[:, None, :] * np.log(self.means) + (1 - X[:, None, :]) * np.log(1 - self.means), axis=-1)

    def _expectation(self, X):
        log_resps = np.log(self.weights) + self._log_bernoulli(X)
        log_resps -= logsumexp(log_resps, axis=-1)[:, None]
        resps = np.exp(log_resps)
        return resps

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.weights = Nk / len(X)
        self.means = (X.T @ resps / Nk).T

    def proba(self, X):
        """
        calculate probability density function
        p(x|theta)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input binary data

        Returns
        -------
        output : (sample_size,) ndarray
            probability
        """
        return np.sum(self.joint_proba(X), axis=-1)

    def joint_proba(self, X):
        """
        calculate joint probability p(X, Z)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input binary data

        Returns
        -------
        joint_prob : (sample_size, n_components) ndarray
            joint probability of the input and the latent variable
        """
        return self.weights * self._bernoulli(X)

    def classify(self, X):
        """
        classify input according to posterior of the latent variable

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input binary data

        Returns
        -------
        output : (sample_size,) ndarray
            classified cluster index
        """
        return np.argmax(self.joint_proba(X), axis=1)

    def classify_proba(self, X):
        """
        compute posterior probability of the latent variable
        p(z|x,theta)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            binary input data

        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior probability
        """
        return self._expectation(X)

import numpy as np


class GaussianMixtureDistribution(object):

    def __init__(self, n_component):
        """
        construct mixture of gaussian model

        Parameters
        ----------
        n_component : int
            number of gaussian components
        """
        self.n_component = n_component

    def fit(self, X, iter_max=10):
        """
        maximul likelihood estimation of parameters with EM algorithm

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        ndim : int
            dimensionality of data space
        weight : (n_component,) ndarray
            mixing coefficient of each component
        means : (ndim, n_component) ndarray
            mean of each gaussian component
        covs : (ndim, ndim, n_component) ndarray
            covariance matrix of each gaussian component
        n_iter : int
            number of iterations performed
        """
        self.ndim = np.size(X, 1)
        self.weights = np.ones(self.n_component) / self.n_component
        self.means = np.random.uniform(X.min(), X.max(), (self.ndim, self.n_component))
        self.covs = np.repeat(10 * np.eye(self.ndim), self.n_component).reshape(self.ndim, self.ndim, self.n_component)
        for i in range(iter_max):
            params = np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))
            resps = self._expectation(X)
            self._maximization(X, resps)
            if np.allclose(params, np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))):
                break
        self.n_iter = i + 1

    def _gauss(self, X):
        precisions = np.linalg.inv(self.covs.T).T
        diffs = X[:, :, None] - self.means
        exponents = np.sum(np.einsum('nik,ijk->njk', diffs, precisions) * diffs, axis=1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs.T).T * (2 * np.pi) ** self.ndim)

    def _expectation(self, X):
        resps = self.weights * self._gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.weights = Nk / len(X)
        self.means = X.T @ resps / Nk
        diffs = X[:, :, None] - self.means
        self.covs = np.einsum('nik,njk->ijk', diffs, diffs * np.expand_dims(resps, 1)) / Nk

    def probability(self, X):
        """
        calculate probability density function

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size,) ndarray
            probability
        """
        joint_prob = self.weights * self._gauss(X)
        return np.sum(joint_prob, axis=-1)

    def classify(self, X):
        """
        classify input

        Parameters
        ----------
        X : (sample_size, ndim)
            input

        Returns
        -------
        output : (sample_size,) ndarray
            corresponding cluster index
        """
        joint_prob = self.weights * self.gauss(X)
        return np.argmax(joint_prob, axis=1)

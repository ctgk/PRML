import numpy as np


class GaussianMixtureDistribution(object):

    def __init__(self, n_components):
        """
        construct mixture of gaussian model

        Parameters
        ----------
        n_component : int
            number of gaussian components
        """
        self.n_components = n_components

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
        means : (n_component, ndim) ndarray
            mean of each gaussian component
        covs : (n_component, ndim, ndim) ndarray
            covariance matrix of each gaussian component
        n_iter : int
            number of iterations performed
        """
        self.ndim = np.size(X, 1)
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = np.random.uniform(X.min(), X.max(), (self.n_components, self.ndim))
        self.covs = np.repeat(np.cov(X, rowvar=False) * 10, self.n_components).reshape(self.ndim, self.ndim, self.n_components).transpose(2, 0, 1)
        for i in range(iter_max):
            params = np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))
            resps = self._expectation(X)
            self._maximization(X, resps)
            if np.allclose(params, np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))):
                break
        self.n_iter = i + 1

    def _gauss(self, X):
        precisions = np.linalg.inv(self.covs)
        diff = X[:, None, :] - self.means
        exponents = np.sum(np.einsum('nki,kij->nkj', diff, precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)

    def _expectation(self, X):
        resps = self.weights * self._gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.weights = Nk / len(X)
        self.means = (X.T @ resps / Nk).T
        diffs = X[:, None, :] - self.means
        self.covs = np.einsum('nki,nkj->kij', diffs, diffs * resps[:, :, None]) / Nk[:, None, None]

    def proba(self, X):
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
        joint_prob = self.joint_proba(X)
        return np.sum(joint_prob, axis=-1)

    def joint_proba(self, X):
        """
        calculate joint probability p(X, Z)

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        joint_prob : (sample_size, n_components) ndarray
            joint probability of input and component
        """
        return self.weights * self._gauss(X)

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
        joint_prob = self.joint_proba(X)
        return np.argmax(joint_prob, axis=1)

    def classify_proba(self, X):
        """
        posterior probability of cluster
        p(z|x,theta)

        Parameters
        ----------
        X : (sample_size, ndim)
            input

        Returns
        -------
        output : (sample_size, n_clus) ndarray
            posterior probability of cluster
        """
        return self._expectation(X)

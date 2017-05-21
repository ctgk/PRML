import numpy as np
from prml.random.random import RandomVariable


class GaussianMixture(RandomVariable):

    def __init__(self, n_components=3, means=None, variances=None, precisions=None, coefs=None):
        """
        construct mixture of Gaussians

        Parameters
        ----------
        n_components : int
            number of gaussian component
        means : (n_components, ndim) np.ndarray
            mean parameter of each gaussian component
        variances : (n_components, ndim, ndim) np.ndarray
            variance parameter of each gaussian component
        precisions : (n_components, ndim, ndim) np.ndarray
            precision parameter of each gaussian component
        coefs : (n_components,) np.ndarray
            mixing coefficients
        """
        assert isinstance(n_components, int)
        assert means is None or isinstance(means, (np.ndarray))
        assert variances is None or isinstance(variances, (int, float, np.ndarray))
        assert precisions is None or isinstance(precisions, (int, float, np.ndarray))
        assert coefs is None or isinstance(coefs, (np.ndarray))
        if means is not None:
            self.n_components = np.size(means, 0)
        else:
            self.n_components = n_components
        self.means = means
        if variances is not None:
            self.variances = variances
        elif precisions is not None:
            self.precisions = precisions
        else:
            self.variances = None
        self.coefs = coefs

    def __setattr__(self, name, value):
        if name is "means":
            if isinstance(value, np.ndarray):
                assert value.ndim == 2
                assert np.size(value, 0) == self.n_components
                object.__setattr__(self, "ndim", np.size(value, 1))
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, None)
        elif name is "variances":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, np.array([np.eye(self.ndim) * value for _ in range(self.n_components)]))
                object.__setattr__(self, "precision", 1 / self.variances)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.n_components, self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "precision", np.linalg.inv(value))
            else:
                object.__setattr__(self, name, None)
        elif name is "precisions":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, np.array([np.eye(self.ndim) * value for _ in range(self.n_components)]))
                object.__setattr__(self, "variances", 1 / self.precisions)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.n_components, self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "variances", np.linalg.inv(value))
        elif name is "coefs":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert np.allclose(value.sum(), 1)
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, np.ones(self.n_components) / self.n_components)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return "GaussianMixture(\nmeans=\n{},\nvariances=\n{}\n,\ncoefs={}\n)".format(self.means, self.variances, self.coefs)

    def _gauss(self, X):
        precisions = np.linalg.inv(self.variances)
        diff = X[:, None, :] - self.means
        exponents = np.sum(np.einsum('nki,kij->nkj', diff, precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.variances) * (2 * np.pi) ** self.ndim)

    def _ml(self, X):
        mean = np.mean(X, axis=0)
        var = np.cov(X, rowvar=False)
        self.means = np.random.multivariate_normal(mean, var * 3, size=self.n_components)
        self.variances = np.array([var for _ in range(self.n_components)])
        self.coefs = np.ones(self.n_components) / self.n_components
        while True:
            params = np.hstack((self.means.ravel(), self.variances.ravel(), self.coefs.ravel()))
            stats = self._expectation(X)
            self._maximization(X, stats)
            if np.allclose(params, np.hstack((self.means.ravel(), self.variances.ravel(), self.coefs.ravel()))):
                break

    def _expectation(self, X):
        resps = self.coefs * self._gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.coefs = Nk / len(X)
        self.means = (X.T @ resps / Nk).T
        diffs = X[:, None, :] - self.means
        self.variances = np.einsum('nki,nkj->kij', diffs, diffs * resps[:, :, None]) / Nk[:, None, None]

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
        return self.coefs * self._gauss(X)

    def _proba(self, X):
        joint_prob = self.joint_proba(X)
        return np.sum(joint_prob, axis=-1)

    def classify(self, X):
        """
        classify input
        max_z p(z|x, theta)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
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
        X : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior probability of cluster
        """
        return self._expectation(X)

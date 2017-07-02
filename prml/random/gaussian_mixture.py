import numpy as np
from prml.random.random import RandomVariable


class GaussianMixture(RandomVariable):
    """
    p(x|mu(means),L(precisions),pi(coefs))
    = sum_k pi_k N(x|mu_k, L_k^-1)
    """

    def __init__(self,
                 n_components=3,
                 means=None,
                 variances=None,
                 precisions=None,
                 coefs=None):
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
                self.ndim = np.size(value, 1)
                object.__setattr__(self, name, value)
            else:
                assert value is None, "means must be either np.ndarray or None"
                object.__setattr__(self, name, None)
        elif name is "variances":
            if isinstance(value, (int, float)):
                object.__setattr__(
                    self, name,
                    value * np.array(
                        [np.eye(self.ndim) for _ in range(self.n_components)]
                    )
                )
                object.__setattr__(self, "precisions", 1 / self.variances)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.n_components, self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "precisions", np.linalg.inv(value))
            else:
                assert value is None, (
                    "variances must be either int, float, np.ndarray, or None"
                )
                object.__setattr__(self, name, None)
        elif name is "precisions":
            if isinstance(value, (int, float)):
                object.__setattr__(
                    self, name,
                    value * np.array(
                        [np.eye(self.ndim) for _ in range(self.n_components)]
                    )
                )
                object.__setattr__(self, "variances", 1 / self.precisions)
            elif isinstance(value, np.ndarray):
                assert value.shape == (self.n_components, self.ndim, self.ndim)
                np.linalg.cholesky(value)
                object.__setattr__(self, name, value)
                object.__setattr__(self, "variances", np.linalg.inv(value))
            else:
                assert value is None, (
                    "precision must be either int, float, np.ndarray, or None"
                )
        elif name is "coefs":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert np.allclose(value.sum(), 1)
                object.__setattr__(self, name, value)
            else:
                assert value is None, "coefs must be either np.ndarray or None"
                object.__setattr__(
                    self, name, np.ones(self.n_components) / self.n_components)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return (
            "GaussianMixture"
            "(\nmeans=\n{},\nvariances=\n{},\ncoefs={}\n)"
            .format(self.means, self.variances, self.coefs)
        )

    @property
    def shape(self):
        if hasattr(self.mean, "shape"):
            return self.means.shape[1:]
        else:
            return None

    @property
    def mean(self):
        return np.sum(self.coefs[:, None] * self.means, axis=0)

    def _gauss(self, X):
        d = X[:, None, :] - self.means
        D_sq = np.sum(np.einsum('nki,kij->nkj', d, self.precisions) * d, -1)
        return (
            np.exp(-0.5 * D_sq)
            / np.sqrt(
                np.linalg.det(self.variances) * (2 * np.pi) ** self.ndim
            )
        )

    def _ml(self, X):
        mean = np.mean(X, axis=0)
        var = np.cov(X.T)
        self.means = np.random.multivariate_normal(
            mean, var * 3, size=self.n_components)
        self.variances = np.array([var for _ in range(self.n_components)])
        self.coefs = np.ones(self.n_components) / self.n_components
        params = np.hstack(
            (self.means.ravel(),
             self.variances.ravel(),
             self.coefs.ravel())
        )
        while True:
            stats = self._expectation(X)
            self._maximization(X, stats)
            new_params = np.hstack(
                (self.means.ravel(),
                 self.variances.ravel(),
                 self.coefs.ravel())
            )
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        resps = self.coefs * self._gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.coefs = Nk / len(X)
        self.means = (X.T @ resps / Nk).T
        d = X[:, None, :] - self.means
        self.variances = np.einsum(
            'nki,nkj->kij', d, d * resps[:, :, None]) / Nk[:, None, None]

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

    def _pdf(self, X):
        joint_prob = self.coefs * self._gauss(X)
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
        return np.argmax(self.classify_proba(X), axis=1)

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

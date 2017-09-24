import numpy as np
from prml.random.random import RandomVariable


class MultivariateGaussianMixture(RandomVariable):
    """
    p(x|mu(means),L(precisions),pi(coefs))
    = sum_k pi_k N(x|mu_k, L_k^-1)
    """

    def __init__(self,
                 n_components=None,
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
            assert n_components is None
        else:
            assert n_components is not None
            self.n_components = n_components
        self.means = means
        if variances is not None:
            assert precisions is None
            self.variances = variances
        elif precisions is not None:
            self.precisions = precisions
        else:
            self.variances = None
        self.coefs = coefs

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, means):
        if isinstance(means, np.ndarray):
            assert means.ndim == 2
            assert np.size(means, 0) == self.n_components
            self.ndim = np.size(means, 1)
            self._means = means
        elif means is None:
            self._means = None
        else:
            raise TypeError("means must be either np.ndarray or None")

    @property
    def variances(self):
        return self._variances

    @variances.setter
    def variances(self, variances):
        if isinstance(variances, (int, float)):
            self._variances = variances * np.array(
                [np.eye(self.ndim) for _ in range(self.n_components)]
            )
            self._precisions = 1 / self._variances
        elif isinstance(variances, np.ndarray):
            assert variances.shape == (self.n_components, self.ndim, self.ndim)
            np.linalg.cholesky(variances)
            self._variances = variances
            self._precisions = np.linalg.inv(variances)
        elif variances is None:
            self._variances = None
        else:
            raise TypeError("variances must be one of these: int, float, np.ndarray, or None")

    @property
    def precisions(self):
        return self._precisions

    @precisions.setter
    def precisions(self, precisions):
        if isinstance(precisions, (int, float)):
            self._precisions = precisions * np.array(
                [np.eye(self.ndim) for _ in range(self.n_components)]
            )
            self._variances = 1 / self._precisions
        elif isinstance(precisions, np.ndarray):
            assert precisions.shape == (self.n_components, self.ndim, self.ndim)
            np.linalg.cholesky(precisions)
            self._precisions = precisions
            self._variances = np.linalg.inv(precisions)
        elif precisions is None:
            self._precisions = None
        else:
            raise TypeError("precisions must be either int, float, np.ndarray, or None")

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, coefs):
        if isinstance(coefs, np.ndarray):
            assert coefs.ndim == 1
            assert np.allclose(coefs.sum(), 1)
            self._coefs = coefs
        elif coefs is None:
            self._coefs = None
        else:
            raise TypeError("coefs must be either np.ndarray or None")

    def __repr__(self):
        return (
            "MultivariateGaussianMixture"
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

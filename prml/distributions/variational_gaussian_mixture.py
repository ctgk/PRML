import numpy as np
from scipy.misc import logsumexp
from scipy.special import digamma, gamma


class VariationalGaussianMixtureDistribution(object):

    def __init__(self, n_components=1, alpha0=None, m0=None, W0=1., dof0=None, beta0=1.):
        """
        construct variational gaussian mixture model

        Parameters
        ----------
        n_components : int
            maximum numnber of gaussian components
        alpha0 : float
            parameter of prior dirichlet distribution
        m0 : float
            mean parameter of prior gaussian distribution
        W0 : float
            mean of the prior Wishart distribution
        dof0 : float
            number of degrees of freedom of the prior Wishart distribution
        beta0 : float
            prior on the precision distribution
        """
        self.n_components = n_components
        if alpha0 is None:
            self.alpha0 = 1 / n_components
        else:
            self.alpha0 = alpha0
        self.m0 = m0
        self.W0 = W0
        self.dof0 = dof0
        self.beta0 = beta0

    def _init_params(self, X):
        sample_size, self.ndim = X.shape
        self.alpha0 = np.ones(self.n_components) * self.alpha0
        if self.m0 is None:
            self.m0 = np.mean(X, axis=0)
        else:
            self.m0 = np.zeros(self.ndim) + self.m0
        self.W0 = np.eye(self.ndim) * self.W0
        if self.dof0 is None:
            self.dof0 = self.ndim

        self.component_size = sample_size / self.n_components + np.zeros(self.n_components)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(sample_size, self.n_components, replace=False)
        self.means = X[indices]
        self.W = np.tile(self.W0, (self.n_components, 1, 1))
        self.dof = self.dof0 + self.component_size

    def get_params(self):
        return self.alpha, self.beta, self.means, self.W, self.dof

    def fit(self, X, iter_max=100):
        """
        perform variational bayesian estimation of parameters
        computes variational posterior distribution q(params|X)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input data
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        ndim : int
            dimensionality of data space
        component_size : float
            sum of responsibilities of each component
        alpha : (n_components,) ndarray
            parameter of posterior dirichlet distribution
        means : (n_components, ndim) ndarray
            mean parameter of posterior gaussian distribution
        beta : float
            precision-like parameter of posterior gaussian distribution
        W : (n_components, ndim, ndim) ndarray
            a parameter of posterior Wishart distribution
        dof : float
            number of degrees of freedom of posterior Wishart distribution
        """
        self._init_params(X)
        for i in range(iter_max):
            params = np.hstack([param.flatten() for param in self.get_params()])
            r = self._variational_expectation(X)
            self._variational_maximization(X, r)
            if np.allclose(params, np.hstack([array.ravel() for array in self.get_params()])):
                break
        else:
            print("parameters may not have converged")

    def _variational_expectation(self, X):
        d = X[:, None, :] - self.means
        maha_sq = -0.5 * (
            self.ndim / self.beta
            + self.dof * np.sum(
                np.einsum("kij,nkj->nki", self.W, d) * d, axis=-1))
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        ln_r = ln_pi + 0.5 * ln_Lambda + maha_sq
        ln_r -= logsumexp(ln_r, axis=-1)[:, None]
        r = np.exp(ln_r)
        return r

    def _variational_maximization(self, X, r):
        self.component_size = r.sum(axis=0)
        Xm = (X.T.dot(r) / self.component_size).T
        d = X[:, None, :] - Xm
        S = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / self.component_size[:, None, None]
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.means = (self.beta0 * self.m0 + self.component_size[:, None] * Xm) / self.beta[:, None]
        d = Xm - self.m0
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S.T).T
            + (self.beta0 * self.component_size * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.component_size)).T)
        self.dof = self.dof0 + self.component_size

    def classify(self, X):
        """
        index of highest posterior of the latent variable

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size, n_components) ndarray
            index of maximum posterior of the latent variable
        """
        return np.argmax(self._variational_expectation(X), 1)

    def classify_proba(self, X):
        """
        compute posterior of the latent variable

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior of the latent variable
        """
        return self._variational_expectation(X)

    def student_t(self, X):
        nu = self.dof + 1 - self.ndim
        L = (nu * self.beta * self.W.T / (1 + self.beta)).T
        d = X[:, None, :] - self.means
        maha_sq = np.sum(np.einsum('nki,kij->nkj', d, L) * d, axis=-1)
        return (
            gamma(0.5 * (nu + self.ndim))
            * np.sqrt(np.linalg.det(L))
            * (1 + maha_sq / nu) ** (-0.5 * (nu + self.ndim))
            / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * self.ndim)))

    def proba(self, X):
        """
        compute predictive distribution p(x|training_data)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input data

        Returns
        -------
        output : (sample_size,) ndarray
            predictive distribution
        """
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()

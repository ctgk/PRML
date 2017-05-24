import numpy as np
from scipy.misc import logsumexp
from prml.random.random import RandomVariable


class BernoulliMixture(RandomVariable):
    """
    p(x|pi,mu)
    = sum_k pi_k mu_k^x (1 - mu_k)^(1 - x)
    """

    def __init__(self, n_components=3, probs=None, coefs=None):
        """
        construct mixture of Bernoulli

        Parameters
        ----------
        n_components : int
            number of bernoulli component
        probs : (n_components, ndim) np.ndarray
            probability of value 1 for each component
        coefs : (n_components,) np.ndarray
            mixing coefficients
        """
        assert isinstance(n_components, int)
        if probs is not None:
            self.n_components = np.size(probs, 0)
        else:
            self.n_components = n_components
        self.probs = probs
        self.coefs = coefs

    def __setattr__(self, name, value):
        if name is "probs":
            if isinstance(value, np.ndarray):
                assert value.ndim == 2
                assert np.size(value, 0) == self.n_components
                assert (value >= 0.).all() and (value <= 1.).all()
                self.ndim = np.size(value, 1)
                object.__setattr__(self, name, value)
            else:
                assert value is None, "probs must be either np.ndarray or None"
                object.__setattr__(self, name, None)
        elif name is "coefs":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                assert np.allclose(value.sum(), 1)
                object.__setattr__(self, name, value)
            else:
                assert value is None, "coefs must be either np.ndarray or None"
                object.__setattr__(self, name, np.ones(self.n_components) / self.n_components)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return (
            "BernoulliMixture"
            "(\nprobs=\n{0.probs},\ncoefs={0.coefs}\n)"
            .format(self)
        )

    @property
    def mean(self):
        return np.sum(self.coefs[:, None] * self.means, 0)

    def _log_bernoulli(self, X):
        np.clip(self.probs, 1e-10, 1 - 1e-10, out=self.probs)
        return (
            X[:, None, :] * np.log(self.probs)
            + (1 - X[:, None, :]) * np.log(1 - self.probs)
        ).sum(axis=-1)

    def _ml(self, X):
        self.probs = np.random.uniform(0.25, 0.75, size=(self.n_components, np.size(X, 1)))
        params = np.hstack((self.probs.ravel(), self.coefs.ravel()))
        while True:
            resp = self._expectation(X)
            self._maximization(X, resp)
            new_params = np.hstack((self.probs.ravel(), self.coefs.ravel()))
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        log_resps = np.log(self.coefs) + self._log_bernoulli(X)
        log_resps -= logsumexp(log_resps, axis=-1)[:, None]
        resps = np.exp(log_resps)
        return resps

    def _maximization(self, X, resp):
        Nk = np.sum(resp, axis=0)
        self.coefs = Nk / len(X)
        self.probs = (X.T @ resp / Nk).T

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

    def classfiy_proba(self, X):
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

import numpy as np
from scipy.special import logsumexp
from prml.rv.rv import RandomVariable


class BernoulliMixture(RandomVariable):
    """
    p(x|pi,mu)
    = sum_k pi_k mu_k^x (1 - mu_k)^(1 - x)
    """

    def __init__(self, n_components=3, mu=None, coef=None):
        """
        construct mixture of Bernoulli

        Parameters
        ----------
        n_components : int
            number of bernoulli component
        mu : (n_components, ndim) np.ndarray
            probability of value 1 for each component
        coef : (n_components,) np.ndarray
            mixing coefficients
        """
        super().__init__()
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.mu = mu
        self.coef = coef

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 2
            assert np.size(mu, 0) == self.n_components
            assert (mu >= 0.).all() and (mu <= 1.).all()
            self.ndim = np.size(mu, 1)
            self.parameter["mu"] = mu
        else:
            assert mu is None
            self.parameter["mu"] = None

    @property
    def coef(self):
        return self.parameter["coef"]

    @coef.setter
    def coef(self, coef):
        if isinstance(coef, np.ndarray):
            assert coef.ndim == 1
            assert np.allclose(coef.sum(), 1)
            self.parameter["coef"] = coef
        else:
            assert coef is None
            self.parameter["coef"] = np.ones(self.n_components) / self.n_components

    def _log_bernoulli(self, x):
        np.clip(self.mu, 1e-10, 1 - 1e-10, out=self.mu)
        return (
            x[:, None, :] * np.log(self.mu)
            + (1 - x[:, None, :]) * np.log(1 - self.mu)
        ).sum(axis=-1)

    def _fit(self, x):
        self.mu = np.random.uniform(0.25, 0.75, size=(self.n_components, np.size(x, 1)))
        params = np.hstack((self.mu.ravel(), self.coef.ravel()))
        while True:
            resp = self._expectation(x)
            self._maximization(x, resp)
            new_params = np.hstack((self.mu.ravel(), self.coef.ravel()))
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, x):
        log_resps = np.log(self.coef) + self._log_bernoulli(x)
        log_resps -= logsumexp(log_resps, axis=-1)[:, None]
        resps = np.exp(log_resps)
        return resps

    def _maximization(self, x, resp):
        Nk = np.sum(resp, axis=0)
        self.coef = Nk / len(x)
        self.mu = (x.T @ resp / Nk).T

    def classify(self, x):
        """
        classify input
        max_z p(z|x, theta)

        Parameters
        ----------
        x : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size,) ndarray
            corresponding cluster index
        """
        return np.argmax(self.classify_proba(x), axis=1)

    def classfiy_proba(self, x):
        """
        posterior probability of cluster
        p(z|x,theta)

        Parameters
        ----------
        x : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior probability of cluster
        """
        return self._expectation(x)

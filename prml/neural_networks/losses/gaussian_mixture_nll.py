import numpy as np
from .loss import Loss


class GaussianMixtureNLL(Loss):
    """
    Negative Log Likelihood of Mixture of Gaussian model
    """

    def __init__(self, n_components):
        """
        set number of gaussian components
        Parameters
        ----------
        n_component : int
            number of gaussian components
        """
        self.n_components = n_components

    def __call__(self, x, t):
        """
        Negative log likelihood of mixture of gaussian with given input
        Parameters
        ----------
        x : ndarray (sample_size, 3 * n_components)
            input
        t : ndarray (sample_size, 1)
            corresponding target data
        Returns
        -------
        output : float
            negative log likelihood of mixture of gaussian
        """
        assert np.size(t, 1) == 1
        sigma, weight, mu = self.forward(x)
        gauss = self.gauss(mu, sigma, t)
        return -np.sum(np.log(np.sum(weight * gauss, axis=1)))

    def gauss(self, mu, sigma, targets):
        """
        gauss function
        Parameters
        ----------
        mu : ndarray (sample_size, n_components)
            mean of each gaussian component
        sigma : ndarray (sample_size, n_components)
            standard deviation of each gaussian component
        targets : ndarray (sample_size, 1)
            corresponding target data
        Returns
        -------
        output : ndarray (sample_size, n_components)
            gaussian
        """
        return np.exp(-0.5 * (mu - targets) ** 2 / np.square(sigma)) / np.sqrt(2 * np.pi * np.square(sigma))

    def forward(self, x):
        """
        compute parameters of mixture of gaussian model

        Parameters
        ----------
        x : ndarray (sample_size, 3 * n_components)
            input

        Returns
        -------
        sigma : ndaray (sample_size, n_components)
            standard deviation of each gaussian component
        weight : ndarray (sample_size, n_components)
            mixing coefficients of mixture of gaussian model
        mu : ndarray (sample_size, n_components)
            mean of each gaussian component
        """
        assert np.size(x, 1) == 3 * self.n_components
        x_sigma, x_weight, mu = np.split(x, [self.n_components, 2 * self.n_components], axis=1)
        sigma = np.exp(x_sigma)
        weight = np.exp(x_weight - np.max(x_weight, 1, keepdims=True))
        weight /= np.sum(weight, axis=1, keepdims=True)
        return sigma, weight, mu

    def backward(self, x, t):
        """
        compute input errors
        Parameters
        ----------
        X : ndarray (sample_size, 3 * n_components)
            input
        targets : ndarray (sample_size, 1)
            corresponding target data
        Returns
        -------
        delta : ndarray (sample_size, 3 * n_components)
            input errors
        """
        assert np.size(t, 1) == 1
        sigma, weight, mu = self.forward(x)
        var = np.square(sigma)
        gamma = weight * self.gauss(mu, sigma, t)
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        delta_mu = gamma * (mu - t) / var
        delta_sigma = gamma * (1 - (mu - t) ** 2 / var)
        delta_weight = weight - gamma
        delta = np.hstack([delta_sigma, delta_weight, delta_mu])
        return delta

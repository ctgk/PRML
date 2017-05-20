import numpy as np
from scipy.misc import logsumexp
from .state_space_model import StateSpaceModel


class Particle(StateSpaceModel):
    """
    A class to perform particle filtering or smoothing
    """

    def __init__(self, n_particles, sigma, ndim, nll=None):
        """
        construct state space model to perform particle filtering or smoothing

        Parameters
        ----------
        n_particles : int
            number of particles
        sigma : int or float
            standard deviation of gaussian transition
        ndim : int
            dimensionality
        nll : callable
            negative log likelihood
        """

        self.n_particles = n_particles
        self.sigma = sigma
        self.ndim = ndim
        if nll is None:
            def nll(obs, particle):
                return np.sum((obs - particle) ** 2, axis=-1)
        self.nll = nll

    def likelihood(self, X, particle):
        logit = -self.nll(X, particle)
        logit -= logsumexp(logit)
        weight = np.exp(logit)
        assert np.allclose(weight.sum(), 1.), weight.sum()
        assert weight.shape == (len(particle),), weight.shape
        return weight

    def resample(self, particle, weight):
        index = np.random.choice(len(particle), size=len(particle), p=weight)
        return particle[index]

    def filtering(self, seq):
        """
        particle filtering
        1. prediction
            p(z_n+1|x_1:n) = \int p(z_n+1|z_n)p(z_n|x_1:n)dz_n
        2. filtering
            p(z_n+1|x_1:n+1) \propto p(x_n+1|z_n+1)p(z_n+1|x_1:n)

        Parameters
        ----------
        seq : (N, ndim_observe) np.ndarray
            observed sequence

        Returns
        -------
        output : type
            explanation of the output
        """
        self.position = []
        position = np.random.normal(size=(self.n_particles, self.ndim))
        for obs in seq:
            delta = np.random.normal(
                scale=self.sigma,
                size=(self.n_particles, self.ndim)
            )
            position = position + delta
            weight = self.likelihood(obs, position)
            position = self.resample(position, weight)
            self.position.append(position)
        self.position = np.asarray(self.position)
        return self.position.mean(axis=1)

    def smoothing(self):
        pass

import numpy as np
from prml.distributions import GaussianDistribution


class GaussianHMM(object):
    """
    Hidden Markov Model with Gaussian emission model
    """

    def __init__(self, initial_proba, transition_proba, means, covs):
        """
        construct hidden markov model with Gaussian emission model

        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray or None
            probability of initial states
        transition_proba : (n_hidden, n_hidden) np.ndarray or None
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        means : (n_hidden, ndim) np.ndarray
            mean of each gaussian component
        covs : (n_hidden, ndim, ndim) np.ndarray
            covariance matrix of each gaussian component

        Attributes
        ----------
        ndim : int
            dimensionality of observation space
        n_hidden : int
            number of hidden states
        """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0] == covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        self.n_hidden = initial_proba.size
        self.ndim = means.shape[1]
        self.initial_proba = initial_proba
        self.transition_proba = transition_proba
        self.means = means
        self.covs = covs
        self.precisions = np.linalg.inv(self.covs)
        self.gaussians = [GaussianDistribution(m, cov) for m, cov in zip(means, covs)]

    def _gauss(self, X):
        diff = X[:, None, :] - self.means
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)

    def draw(self, n=100):
        """
        draw random sequence from this model

        Parameters
        ----------
        n : int
            length of the random sequence

        Returns
        -------
        seq : (n, ndim) np.ndarray
            generated random sequence
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.extend(self.gaussians[hidden_state].draw())
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def forward_backward(self, seq):
        """
        estimate posterior distributions of hidden states

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence

        Returns
        -------
        posterior : (N, n_hidden) np.ndarray
            posterior distribution of hidden states
        """
        N = len(seq)
        likelihood = self._gauss(seq)
        forward = [self.initial_proba * likelihood[0]]
        backward = [likelihood[-1]]
        for i in range(1, N):
            forward.append(self.transition_proba @ forward[-1] * likelihood[i])
        for i in range(N - 2, -1, -1):
            backward.insert(0, (likelihood[i] * backward[0]) @ self.transition_proba)
        forward = np.asarray(forward)
        backward = np.asarray(backward)
        posterior = forward * backward
        posterior /= np.sum(posterior, axis=-1, keepdims=True)
        return posterior

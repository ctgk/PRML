import numpy as np


class CategoricalHMM(object):
    """
    Hidden Markov Model with categorical emission model
    """

    def __init__(self, initial_proba, transition_proba, means):
        """
        construct hidden markov model with categorical emission model

        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray
            probability of initial latent state
        transition_proba : (n_hidden, n_hidden) np.ndarray
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        means : (n_hidden, ndim) np.ndarray
            mean parameters of categorical distribution

        Returns
        -------
        ndim : int
            number of observation categories
        n_hidden : int
            number of hidden states
        """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0]
        assert np.allclose(means.sum(axis=1), 1)
        self.n_hidden = initial_proba.size
        self.ndim = means.shape[1]

        self.initial_proba = initial_proba
        self.transition_proba = transition_proba
        self.means = means

    def draw(self, n=100):
        """
        draw random sequence from this model

        Parameters
        ----------
        n : int
            length of the random sequence

        Returns
        -------
        seq : (n,) np.ndarray
            generated random sequence
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.append(np.random.choice(self.ndim, p=self.means[hidden_state]))
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def forward_backward(self, seq):
        """
        estimate each posterior distribution of hidden state

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence

        Returns
        -------
        posterior : (N, n_hidden) np.ndarray
            posterior distributions of each latent variable
        """
        N = len(seq)
        likelihood = self.means[seq]
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

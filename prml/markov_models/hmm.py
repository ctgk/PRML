import numpy as np


class HiddenMarkovModel(object):
    """
    base class of various hidden markov models
    """

    def __init__(self, n_states=None):
        """
        construct hidden markov model

        Parameters
        ----------
        n_states : int or None
            number of hidden states

        Attributes
        ----------
        initial_proba : (n_states,) ndarray
            probability of initial states of sequence
        transition_proba : (n_states, n_states) ndarray
            transition probability of states of sequence
            (i, j) component denotes the probability from i-th to j-th state
        """
        assert (n_states is None) or isinstance(n_states, int)
        self.n_states = n_states
        if isinstance(n_states, int):
            self.initial_proba = np.ones(n_states) / n_states
            self.transition_proba = np.random.rand(n_states, n_states)
            self.transition_proba /= np.sum(
                self.transition_proba, axis=1, keepdims=True)

    def set_initial_proba(self, initial_proba):
        """
        set probability of the initial state

        Parameters
        ----------
        initial_proba : (n_states,) np.ndarray
            probability of initial state
        """
        assert isinstance(initial_proba, np.ndarray)
        if isinstance(self.n_states, int):
            assert initial_proba.shape == (self.n_states,), initial_proba.shape
        else:
            assert initial_proba.ndim == 1, initial_proba.ndim
            self.n_states = len(initial_proba)
        self.initial_proba = initial_proba

    def set_transition_proba(self, transition_proba, normalize=True):
        """
        set transition probability matrix of hidden states

        Parameters
        ----------
        transition_proba : (n_states, n_states) np.ndarray
            transition probability matrix
            (i, j) component denotes the probability from i-th to j-th state
        normalize : bool
            if True, normalize the probabitliy matrix
        """
        assert isinstance(transition_proba, np.ndarray)
        if isinstance(self.n_states, int):
            assert transition_proba.shape == (self.n_states, self.n_states), transition_proba.shape
        else:
            assert transition_proba.ndim == 2, transition_proba.ndim
            assert len(transition_proba) == len(transition_proba[0]), transition_proba.shape
            self.n_states = len(transition_proba)
        self.transition_proba = transition_proba
        if normalize:
            self.transition_proba /= np.sum(
                self.transition_proba, axis=1, keepdims=True)
        assert np.allclose(1, self.transition_proba.sum(axis=1)), self.transition_proba.sum(axis=1)

    def set_emission_dist(self, emission_dist):
        """
        set emission distribution

        Parameters
        ----------
        emission_dist
            emission distribution
        """
        assert emission_dist.n_components == self.n_states
        self.emission_dist = emission_dist

    def filtering(self, seq):
        """
        bayesian filtering
        p(z_0|x_1) -> ... -> p(z_n|x_1:n) -> p(z_n+1|x_1:n+1) -> ...
        1. p(z_n+1|x_1:n) = \int p(z_n+1|z_n)p(z_n|x_1:n) dz_n
        2. p(z_n+1|x_1:n+1) \propto p(x_n+1|z_n+1)p(z_n+1|x_1:n)

        Parameters
        ----------
        seq : (len_seq, dim) np.ndarray
            observed sequential data

        Attributes
        ----------
        filter_proba : (len_seq, n_states) np.ndarray
            latent posterior probability by bayesian filtering
        """
        # p(x_i|z_i)
        emission_proba = self.emission_dist._gauss(seq)
        p = self.initial_proba * emission_proba[0]
        filter_proba = [p / p.sum()]
        for i in range(1, len(seq)):
            pred_proba = filter_proba[-1] @ self.transition_proba
            p = pred_proba * emission_proba[i]
            filter_proba.append(p / p.sum())
        self.filter_proba = np.asarray(filter_proba)

    def smoothing(self, seq):
        """
        bayesian smoothing
        p(z_N|x_1:N) -> ... -> p(z_n+1|x_1:n+1) -> p(z_n|x_1:n) -> ...

        Parameters
        ----------
        seq : (len_seq, dim) ndarray
            observed sequential data

        Attributes
        ----------
        smooth_proba : (len_seq, n_states) np.ndarray
            latent posterior probability by bayesian smoothing
        """
        pass

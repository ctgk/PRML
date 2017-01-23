import numpy as np
from .loss import Loss


class WeightDecay(Loss):
    """
    l2 regularization of parameters
    """

    def __init__(self, alpha=1e-3):
        """
        construct weight decay cost function

        Parameters
        ----------
        alpha : float
            coefficient to be multiplied with l2 penalty
        """
        assert isinstance(alpha, float)
        self.alpha = alpha

    def __call__(self, network):
        """
        compute l2 penalty of parameters in the network

        Parameters
        ----------
        network : Network
            neural network containing the parameters

        Returns
        -------
        penalty : float
            l2 penalty
        """
        penalty = 0
        for layer in network.trainables:
            penalty += np.sum(np.square(layer.param))
        penalty *= 0.5 * self.alpha
        return penalty

    def backward(self, network):
        """
        compute and add derivative with respect to the parameters

        Parameters
        ----------
        network : Network
            neural network with the parameters
        """
        for layer in network.trainables:
            layer.deriv += self.alpha * layer.param

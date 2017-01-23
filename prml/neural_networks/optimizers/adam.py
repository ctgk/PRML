import numpy as np
from .optimizer import Optimizer


class AdamOptimizer(Optimizer):
    """
    Adam optimizer

    initialization
    m1 = 0 (Initial 1st moment of gradient)
    m2 = 0 (Initial 2nd moment of gradient)
    n_iter = 0

    update rule
    n_iter += 1
    learning_rate *= sqrt(1 - beta2^n) / (1 - beta1^n)
    m1 = beta1 * m1 + (1 - beta1) * gradient
    m2 = beta2 * m2 + (1 - beta2) * gradient^2
    param -= learning_rate * m1 / (sqrt(m2) + epsilon)
    """

    def __init__(self, network, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        construct Adam optimizer

        Parameters
        ----------
        network : Network
            neural network to be optimized
        learning_rate : float
        beta1 : float
            exponential decay rate for the 1st moment
        beta2 : float
            exponentail decay rate for the 2nd moment
        epsilon : float
            small constant to be added to denominator for numerical stability

        Attributes
        ----------
        n_iter : int
            number of iterations performed
        moment1 : dict
            1st moment of each learnable parameter
        moment2 : dict
            2nd moment of each learnable parameter
        """
        super().__init__(network, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = {}
        self.moment2 = {}
        for layer in self.network.trainables:
            self.moment1[layer] = np.zeros_like(layer.param)
            self.moment2[layer] = np.zeros_like(layer.param)

    def update(self):
        """
        update parameters of the neural network
        """
        self.increment_iteration()
        lr = (
            self.learning_rate
            * (1 - self.beta2 ** self.n_iter) ** 0.5
            / (1 - self.beta1 ** self.n_iter))
        for layer in self.network.trainables:
            m1 = self.moment1[layer]
            m2 = self.moment2[layer]
            m1 += (1 - self.beta1) * (layer.deriv - m1)
            m2 += (1 - self.beta2) * (layer.deriv ** 2 - m2)
            layer.param -= lr * m1 / (np.sqrt(m2) + self.epsilon)

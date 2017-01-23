import numpy as np
from .optimizer import Optimizer


class AdaGradOptimizer(Optimizer):
    """
    AdaGrad optimizer

    initialization
    G = 0

    update rule
    G += gradient ** 2
    param -= learning_rate * gradient / sqrt(G + eps)
    """

    def __init__(self, network, learning_rate=0.001, epsilon=1e-8):
        super().__init__(network, learning_rate)
        self.epsilon = epsilon
        self.G = {}
        for layer in self.network.trainables:
            self.G[layer] = np.zeros_like(layer.param)

    def update(self):
        """
        update parameters
        """
        self.increment_iteration()
        for layer in self.network.trainables:
            G = self.G[layer]
            deriv = layer.deriv
            G += deriv ** 2
            layer.param -= self.learning_rate * deriv / (np.sqrt(G) + self.epsilon)

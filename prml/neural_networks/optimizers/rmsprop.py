import numpy as np
from .optimizer import Optimizer


class RMSPropOptimizer(Optimizer):
    """
    RMSProp optimizer

    initial
    msd = 0

    update rule
    msd = rho * msd + (1 - rho) * gradient ** 2
    param -= learning_rate * gradient / (sqrt(msd) + eps)
    """

    def __init__(self, network, learning_rate=1e-3, rho=0.9, epsilon=1e-8):
        super().__init__(network, learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_deriv = {}
        for layer in self.network.trainables:
            self.mean_squared_deriv[layer] = np.zeros_like(layer.param)

    def update(self):
        """
        update parameters
        """
        self.increment_iteration()
        for layer in self.network.trainables:
            msd = self.mean_squared_deriv[layer]
            deriv = layer.deriv
            msd *= self.rho
            msd += (1 - self.rho) * deriv ** 2
            layer.param -= self.learning_rate * deriv / (np.sqrt(msd) + self.epsilon)

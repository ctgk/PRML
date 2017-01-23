import numpy as np
from .optimizer import Optimizer


class AdaDeltaOptimizer(Optimizer):
    """
    AdaDelta optimizer
    """

    def __init__(self, network, rho=0.95, epsilon=1e-8):
        super().__init__(network, None)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_deriv = {}
        self.mean_squared_update = {}
        for layer in self.network.trainables:
            self.mean_squared_deriv[layer] = np.zeros_like(layer.param)
            self.mean_squared_update[layer] = np.zeros_like(layer.param)

    def update(self):
        self.increment_iteration()
        for layer in self.network.trainables:
            deriv = layer.deriv
            msd = self.mean_squared_deriv[layer]
            msu = self.mean_squared_update[layer]

            msd *= self.rho
            msd += (1 - self.rho) * deriv ** 2
            delta = np.sqrt((msu + self.epsilon) / (msd + self.epsilon)) * deriv
            msu *= self.rho
            msu *= (1 - self.rho) * delta ** 2
            layer.param -= delta

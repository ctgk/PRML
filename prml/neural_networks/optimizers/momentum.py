import numpy as np
from .optimizer import Optimizer


class MomentumOptimizer(Optimizer):
    """
    Momentum optimizer

    initialization
    v = 0

    update rule
    v = v * momentum - learning_rate * gradient
    param += v
    """

    def __init__(self, network, learning_rate, momentum=0.9):
        super().__init__(network, learning_rate)
        self.momentum = momentum
        self.inertia = {}
        for layer in self.network.trainables:
            self.inertia[layer] = np.zeros_like(layer.param)

    def update(self):
        self.increment_iteration()
        for layer in self.network.trainables:
            inertia = self.inertia[layer]
            inertia *= self.momentum
            inertia -= self.learning_rate * layer.deriv
            layer.param += inertia

import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class Momentum(Optimizer):
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
        for key, param in self.params.items():
            self.inertia[key] = np.zeros(param.shape)

    def update(self):
        self.increment_iteration()
        for key, param in self.params.items():
            inertia = self.inertia[key]
            inertia *= self.momentum
            inertia -= self.learning_rate * param.grad
            param.value += inertia

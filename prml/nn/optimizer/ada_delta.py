import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class AdaDelta(Optimizer):
    """
    AdaDelta optimizer
    """

    def __init__(self, network, rho=0.95, epsilon=1e-8):
        super().__init__(network, None)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_deriv = {}
        self.mean_squared_update = {}
        for key, param in self.params.items():
            self.mean_squared_deriv[key] = np.zeros(param.shape)
            self.mean_squared_update[key] = np.zeros(param.shape)

    def update(self):
        self.increment_iteration()
        for key, param in self.params.items():
            grad = param.grad
            msd = self.mean_squared_deriv[key]
            msu = self.mean_squared_update[key]

            msd *= self.rho
            msd += (1 - self.rho) * grad ** 2
            delta = np.sqrt((msu + self.epsilon) / (msd + self.epsilon)) * grad
            msu *= self.rho
            msu *= (1 - self.rho) * delta ** 2
            param.value -= delta

import numpy as np
from prml.nn.config import config
from prml.nn.optimizer.optimizer import Optimizer


class AdaDelta(Optimizer):
    """
    AdaDelta optimizer
    """

    def __init__(self, parameter: dict, rho=0.95, epsilon=1e-8):
        super().__init__(parameter, None)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_deriv = {}
        self.mean_squared_update = {}
        for key, param in self.parameter.items():
            self.mean_squared_deriv[key] = np.zeros(param.shape, dtype=config.dtype)
            self.mean_squared_update[key] = np.zeros(param.shape, dtype=config.dtype)

    def update(self):
        for key in self.parameter:
            param = self.parameter[key]
            if param.grad is None:
                continue
            msd = self.mean_squared_deriv[key]
            msu = self.mean_squared_update[key]
            grad = param.grad
            msd *= self.rho
            msd += (1 - self.rho) * grad ** 2
            delta = np.sqrt((msu + self.epsilon) / (msd + self.epsilon)) * grad
            msu *= self.rho
            msu *= (1 - self.rho) * delta ** 2
            param.value += delta

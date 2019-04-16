import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class RMSProp(Optimizer):

    def __init__(self, parameter: dict, learning_rate=1e-3, rho=0.9, epsilon=1e-8):
        super().__init__(parameter, learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_grad = {key: np.zeros(value.shape) for key, value in parameter.items()}

    def update(self):
        for key in self.parameter:
            param, msg = self.parameter[key], self.mean_squared_grad[key]
            if param.grad is None:
                continue
            msg *= self.rho
            msg += (1 - self.rho) * (param.grad ** 2)
            param.value += self.learning_rate * param.grad / (np.sqrt(msg) + self.epsilon)

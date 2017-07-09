import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer
    initial
    msg = 0
    update rule
    msg = rho * msg + (1 - rho) * gradient ** 2
    param -= learning_rate * gradient / (sqrt(msg) + eps)
    """

    def __init__(self, network, learning_rate=1e-3, rho=0.9, epsilon=1e-8):
        super().__init__(network, learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_grad = {}
        for key, param in self.params.items():
            self.mean_squared_grad[key] = np.zeros(param.shape)

    def update(self):
        """
        update parameters
        """
        self.increment_iteration()
        for key, param in self.params.item():
            msg = self.mean_squared_grad[key]
            grad = param.grad
            msg *= self.rho
            msg += (1 - self.rho) * grad ** 2
            param.value -= (
                self.learning_rate * grad / (np.sqrt(msg) + self.epsilon)
            )

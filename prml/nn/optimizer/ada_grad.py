import numpy as np
from prml.nn.config import config
from prml.nn.optimizer.optimizer import Optimizer


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer
    initialization
    G = 0
    update rule
    G += gradient ** 2
    param -= learning_rate * gradient / sqrt(G + eps)
    """

    def __init__(self, parameter: dict, learning_rate=0.001, epsilon=1e-8):
        super().__init__(parameter, learning_rate)
        self.epsilon = epsilon
        self.G = []
        for key, param in self.parameter.items():
            self.G[key] = np.zeros(param.shape, dtype=config.dtype)

    def update(self):
        """
        update parameters
        """
        for key in self.parameter:
            param, G = self.parameter[key], self.G[key]
            if param.grad is None:
                    continue
            grad = param.grad
            G += grad ** 2
            param.value += self.learning_rate * grad / (np.sqrt(G) + self.epsilon)

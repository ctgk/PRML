import numpy as np
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

    def __init__(self, network, learning_rate=0.001, epsilon=1e-8):
        super().__init__(network, learning_rate)
        self.epsilon = epsilon
        self.G = {}
        for key, param in self.params.items():
            self.G[key] = np.zeros(param.shape)

    def update(self):
        """
        update parameters
        """
        self.increment_iteration()
        for key, param in self.params.items():
            G = self.G[key]
            grad = param.grad
            G += grad ** 2
            param.value -= self.learning_rate * grad / (np.sqrt(G) + self.epsilon)

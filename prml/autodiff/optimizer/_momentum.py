import numpy as np

from prml.autodiff._core._config import config
from prml.autodiff.optimizer._optimizer import _Optimizer


class Momentum(_Optimizer):

    def __init__(self, parameter: dict, learning_rate=1e-3, momentum=0.9):
        super().__init__(parameter, learning_rate)
        self.momentum = momentum
        self.inertia = {
            key: np.zeros(value.shape, dtype=config.dtype)
            for key, value in parameter.items()
        }

    def update(self):
        for key in self.parameter:
            param, inertia = self.parameter[key], self.inertia[key]
            if param.grad is None:
                continue
            inertia *= self.momentum
            inertia += self.learning_rate * (1 - self.momentum) * param.grad
            param.value += inertia

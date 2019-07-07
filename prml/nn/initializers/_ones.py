import numpy as np

from prml.nn.initializers._initializer import Initializer


class Ones(Initializer):

    def _forward(self, size):
        return np.ones(size)

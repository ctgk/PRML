import numpy as np

from prml.nn.initializers._initializer import Initializer


class Zeros(Initializer):

    def _forward(self, size):
        return np.zeros(size)

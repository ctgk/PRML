import numpy as np

from prml import autodiff


class Initializer(object):
    """
    Base initializer class
    """

    def __init__(self, value: float = None):
        self.value = value

    def _forward(self, size):
        return np.broadcast_to(self.value, size)

    def __call__(self, size):
        """
        return initialized array

        Parameters
        ----------
        size: int or tuple of ints
            size of initialized array

        Returns
        -------
        Array
            initialized array
        """
        return autodiff.asarray(self._forward(size))

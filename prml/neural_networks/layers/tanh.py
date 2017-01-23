import numpy as np
from .layer import Layer


class Tanh(Layer):
    """
    Hyperbolic tangent activation

    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    def forward(self, x, training=False):
        """
        element-wise hyperbolic tangent activation

        Parameters
        ----------
        x : ndarray
            input

        Returns
        -------
        output : ndarray
            hyperbolic tangent of each element
        """
        if training:
            self.output = np.tanh(x)
            return self.output
        else:
            return np.tanh(x)

    def backward(self, delta):
        """
        backpropagation of errors

        Parameters
        ----------
        delta : ndarray
            output errors

        Returns
        -------
        output : ndarray
            input errors
        """
        return (1 - np.square(self.output)) * delta

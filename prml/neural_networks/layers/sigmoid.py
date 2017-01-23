import numpy as np
from .layer import Layer


class Sigmoid(Layer):
    """
    Logistic sigmoid transformation

    y = 1 / (1 + exp(-x))
    """

    def forward(self, x, training=False):
        """
        element-wise transformation by logistic sigmoid function

        Parameters
        ----------
        x : ndarray
            input

        Returns
        -------
        output : ndarray
            logistic sigmoid of each element
        """
        if training:
            self.output = np.tanh(x * 0.5) * 0.5 + 0.5
            return self.output
        else:
            return np.tanh(x * 0.5) * 0.5 + 0.5

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
        return self.output * (1 - self.output) * delta

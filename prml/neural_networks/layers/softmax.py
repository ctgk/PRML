import numpy as np
from .layer import Layer


class Softmax(Layer):
    """
    Softmax activation layer
    """

    def __init__(self, axis=-1):
        """
        construct softmax layer

        Parameters
        ----------
        axis : int
            axis to normalize softmax activations along
        """
        self.axis = axis
        self.istrainable = False

    def forward(self, x, training=False):
        if training:
            self.output = np.exp(x - x.max(axis=self.axis, keepdims=True))
            self.output /= np.sum(self.output, axis=self.axis, keepdims=True)
            return self.output
        else:
            y = np.exp(x - x.max(axis=self.axis, keepdims=True))
            y /= np.sum(y, axis=self.axis, keepdims=True)
            return y

    def backward(self, delta):
        delta_in = self.output * delta
        delta_in -= self.output * delta_in.sum(axis=self.axis, keepdims=True)
        return delta_in

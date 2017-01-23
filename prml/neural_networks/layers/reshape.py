import numpy as np
from .layer import Layer


class Reshape(Layer):
    """Reshape array"""

    def __init__(self, shape):
        self.output_shape = shape
        self.istrainable = False

    def forward(self, x, training=False):
        if training:
            self.input_shape = x.shape
        return np.reshape(x, self.output_shape)

    def backward(self, delta):
        return np.reshape(delta, self.input_shape)

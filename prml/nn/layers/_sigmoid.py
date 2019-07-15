from prml import autodiff
from prml.nn.layers._layer import _Layer


class Sigmoid(_Layer):

    def __init__(self):
        super().__init__()

    def _forward(self, x):
        return autodiff.sigmoid(x)

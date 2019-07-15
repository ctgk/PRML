from prml import autodiff
from prml.nn.layers._layer import _Layer


class ReLU(_Layer):

    def __init__(self):
        super().__init__()

    def _forward(self, x):
        return autodiff.relu(x)

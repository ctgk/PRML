from prml import autodiff
from prml.nn.layers._layer import _Layer


class Flatten(_Layer):

    def __init__(
        self,
        activation=None
    ):
        super().__init__(activation, bias=None)

    def _forward(self, x):
        return autodiff.reshape(x, (len(x), -1))

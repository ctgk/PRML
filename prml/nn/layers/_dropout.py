from prml.nn.functions._dropout import _Dropout
from prml.nn.layers._layer import _Layer


class Dropout(_Layer):

    def __init__(
        self,
        droprate: float = 0.5
    ):
        super().__init__()
        self._func = _Dropout(droprate)

    def _forward(self, x, droprate=None, use_dropout=True):
        if use_dropout:
            return self._func.forward(x, droprate=droprate)
        else:
            return x

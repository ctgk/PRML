from prml import autodiff
from prml.nn.layers._layer import _Layer


class Network(autodiff.Module):

    def __init__(self, *args):
        super().__init__()
        self.layers = []
        for arg in args:
            self.append(arg)

    def _get_next_key(self):
        return "layer_" + str(len(self.layers))

    def append(self, layer: _Layer):
        if not isinstance(layer, _Layer):
            raise TypeError
        with self.initialize():
            self.__setattr__(self._get_next_key(), layer)
        self.layers.append(layer)

    def __call__(self, x, index: int = None):
        for layer in self.layers[:index]:
            x = layer(x)
        return x

    def forward(self, x, index: int = None):
        for layer in self.layers[:index]:
            x = layer.forward(x)
        return x

    def forward_deterministic(self, x, index: int = None):
        for layer in self.layers[:index]:
            x = getattr(layer, "forward_deterministic", layer.forward)(x)
        return x

    def loss(self, index: int = None):
        loss_ = 0
        for layer in self.layers[:index]:
            if hasattr(layer, "loss"):
                loss_ = loss_ + layer.loss()
        return loss_

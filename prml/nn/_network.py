from prml import autodiff
from prml.nn.layers._layer import _Layer


class Sequential(autodiff.Module):

    def __init__(self, *args):
        super().__init__()
        self._layers = []
        for arg in args:
            self.append(arg)

    def _get_next_key(self):
        return "layer_" + str(len(self._layers))

    def append(self, layer: _Layer):
        if not isinstance(layer, _Layer):
            raise TypeError
        with self.initialize():
            self.__setattr__(self._get_next_key(), layer)
        self._layers.append(layer)

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

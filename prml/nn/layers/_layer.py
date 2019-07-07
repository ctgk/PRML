from prml import autodiff
from prml.nn.initializers._initializer import Initializer
from prml.nn.initializers._normal import Normal


class _Layer(autodiff.Module):
    """
    Base layer class
    """

    def __init__(self, activation=None, bias=None):
        super().__init__()
        self.activation = activation
        if bias is not None:
            with self.initialize():
                self.bias = bias
        else:
            self.bias = None

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        y = self._forward(*args, **kwargs)
        if self.bias is not None:
            y = y + self.bias
        if self.activation is not None:
            y = self.activation(y)
        return y


class _TrainableLayer(_Layer):
    """
    Base trainable layer class
    """

    def __init__(self, activation=None, initializer=None, bias=None):
        super().__init__(activation=activation, bias=bias)
        self.initializer = (
            Normal(0, 0.05) if initializer is None else initializer)
        if not isinstance(self.initializer, Initializer):
            raise TypeError(
                "initializer must be prml.nn.initializers.Initializer")

import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class WeightDecay(Function):

    def _forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        loss = 0
        for arg in args:
            loss += np.square(arg.value).sum()
        for arg in kwargs.values():
            loss += np.square(arg.value).sum()
        return Tensor(0.5 * loss, function=self)

    def _backward(self, delta):
        for arg in self.args:
            arg.backward(delta * arg.value)
        for arg in self.kwargs.values():
            arg.backward(delta * arg.value)


def weight_decay(*args, **kwargs):
    return WeightDecay().forward(*args, **kwargs)

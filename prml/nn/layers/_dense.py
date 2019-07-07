from prml import autodiff
from prml.nn.initializers._zeros import Zeros
from prml.nn.layers._layer import _TrainableLayer


class Dense(_TrainableLayer):

    def __init__(
        self,
        ndim_in: int,
        ndim_out: int,
        activation=None,
        initializer=None,
        bias=Zeros()
    ):
        super().__init__(activation, initializer, bias(ndim_out))
        with self.initialize():
            self.weight = self.initializer(size=(ndim_in, ndim_out))

    def _forward(self, x):
        return autodiff.matmul(x, self.weight)

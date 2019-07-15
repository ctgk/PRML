from prml import autodiff, bayesnet
from prml.nn.layers._layer import _TrainableLayer, _BayesianLayer


class Dense(_TrainableLayer):

    def __init__(
        self,
        ndim_in: int,
        ndim_out: int,
        initializer=None,
        has_bias: bool = True
    ):
        super().__init__(initializer)
        with self.initialize():
            self.weight = self.initializer(size=(ndim_in, ndim_out))
        if has_bias:
            self.initialize_bias(ndim_out)

    def _forward(self, x):
        return autodiff.matmul(x, self.weight)


class DenseBayesian(_BayesianLayer):

    def __init__(self, ndim_in: int, ndim_out: int, has_bias: bool = True):
        super().__init__()
        with self.initialize():
            self.qweight = bayesnet.Gaussian(
                var="w", name="q", size=(ndim_in, ndim_out))
        if has_bias:
            self.initialize_bias(size=ndim_out)

    def _forward(self, x):
        return autodiff.matmul(x, self.qweight.sample()["w"])

    def _loss(self):
        return bayesnet._kl_divergence.kl_gaussian(
            self.qweight, self.prior)

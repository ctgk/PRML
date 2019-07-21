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
                var="w", size=(ndim_in, ndim_out))
        if has_bias:
            self.initialize_bias(size=ndim_out)

    def _forward(self, x):
        return autodiff.matmul(x, self.qweight.sample()["w"])

    def _loss(self):
        return bayesnet._kl_divergence.kl_gaussian(
            self.qweight, self.prior)


class _Gaussian(bayesnet.Gaussian):

    def __init__(self, var, condition):
        super().__init__(var=var, conditions=[condition])
        self.condition = condition

    def forward(self, **kwargs):
        return {"mean": 0, "std": 1 / autodiff.sqrt(kwargs[self.condition])}


class DenseARD(_BayesianLayer):

    def __init__(self, ndim_in: int, ndim_out: int, has_bias: bool = True):
        super().__init__()
        self.ptau_w = bayesnet.Exponential("tau_w", rate=1e-2)
        self.pw = _Gaussian("w", "tau_w")
        with self.initialize():
            self.qtau_w = bayesnet.Exponential(
                var="tau_w", size=(ndim_in, ndim_out))
            self.qweight = bayesnet.Gaussian(var="w", size=(ndim_in, ndim_out))
        if has_bias:
            self.ptau_b = bayesnet.Exponential("tau_b", rate=1e-2)
            self.pb = _Gaussian("b", "tau_b")
            with self.initialize():
                self.qtau_b = bayesnet.Exponential(
                    var="tau_b", size=ndim_out)
            self.initialize_bias(size=ndim_out)

    def _forward(self, x):
        return autodiff.matmul(x, self.qweight.sample()["w"])

    def loss(self):
        tau_w = self.qtau_w.sample()["tau_w"]
        loss_ = (
            bayesnet._kl_divergence.kl_gaussian(self.qweight, self.pw, tau_w=tau_w)
            + bayesnet._kl_divergence.kl_exponential(self.qtau_w, self.ptau_w)
        )
        if self.qbias is not None:
            tau_b = self.qtau_b.sample()["tau_b"]
            loss_ = (
                loss_
                + bayesnet._kl_divergence.kl_gaussian(
                    self.qbias, self.pb, tau_b=tau_b)
                + bayesnet._kl_divergence.kl_exponential(
                    self.qtau_b, self.ptau_b)
            )
        return loss_

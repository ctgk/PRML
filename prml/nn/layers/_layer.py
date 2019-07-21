from prml import autodiff, bayesnet
from prml.nn.initializers._initializer import Initializer
from prml.nn.initializers._normal import Normal


class _Layer(autodiff.Module):
    """
    Base layer class
    """

    def __init__(self):
        super().__init__()
        self.bias = None

    def initialize_bias(self, size: int or tuple):
        with self.initialize():
            self.bias = autodiff.zeros(size)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        y = self._forward(*args, **kwargs)
        if self.bias is not None:
            y = y + self.bias
        return y


class _TrainableLayer(_Layer):
    """
    Base trainable layer class
    """

    def __init__(self, initializer=None):
        super().__init__()
        self.initializer = (
            Normal(0, 0.05) if initializer is None else initializer)
        if not isinstance(self.initializer, Initializer):
            raise TypeError(
                "initializer must be prml.nn.initializers.Initializer")


class _BayesianLayer(_Layer):
    """
    Base Bayesian layer class
    """

    def __init__(self):
        super().__init__()
        self.prior = bayesnet.Gaussian(mean=0, std=1)
        self.qbias = None

    def initialize_bias(self, size: int or tuple):
        with self.initialize():
            self.qbias = bayesnet.Gaussian(var="b", size=size)

    def __call__(self, *args, **kwargs):
        y = self._forward(*args, **kwargs)
        if self.qbias is not None:
            y = y + self.qbias.sample()["b"]
        return y

    def _loss(self):
        raise NotImplementedError

    def loss(self):
        loss_ = self._loss()
        if self.qbias is not None:
            loss_ = loss_ + bayesnet._kl_divergence.kl_gaussian(
                self.qbias, self.prior)
        return loss_

import numpy as np
from prml.tensor.tensor import Tensor
from prml.function.function import Function


class MultivariateGaussian(Function):
    """
    multivariate gaussian sampling
    """

    def _check_input(self, mean, cov):
        mean = self._convert2tensor(mean)
        cov = self._convert2tensor(cov)
        self._equal_ndim(mean, 1)
        self._equal_ndim(cov, 2)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(
                "covariance must be square matrix, shape {} is not square".format(cov.shape)
            )
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(
                "shapes {} and {} not aligned: {} (dim 0) != {} (dim 0, 1)"
                .format(mean.shape, cov.shape, mean.shape[0], cov.shape[0])
            )
        return mean, cov

    def _forward(self, mean, cov):
        mean, cov = self._check_input(mean, cov)
        self.mean = mean
        self.cov = cov
        self.eps = np.random.normal(size=mean.shape)
        output = mean.value + np.linalg.cholesky(cov.value) @ self.eps
        return Tensor(output, function=self)

    def _backward(self, delta):
        raise NotImplementedError

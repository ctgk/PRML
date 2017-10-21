from prml.autograd.math.add import add
from prml.autograd.math.divide import divide, rdivide
from prml.autograd.math.exp import exp
from prml.autograd.math.log import log
from prml.autograd.math.matmul import matmul, rmatmul
from prml.autograd.math.mean import mean
from prml.autograd.math.multiply import multiply
from prml.autograd.math.negative import negative
from prml.autograd.math.power import power, rpower
from prml.autograd.math.product import prod
from prml.autograd.math.sqrt import sqrt
from prml.autograd.math.square import square
from prml.autograd.math.subtract import subtract, rsubtract
from prml.autograd.math.sum import sum


from prml.autograd.tensor.tensor import Tensor
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__truediv__ = divide
Tensor.__rtruediv__ = rdivide
Tensor.mean = mean
Tensor.__matmul__ = matmul
Tensor.__rmatmul__ = rmatmul
Tensor.__mul__ = multiply
Tensor.__rmul__ = multiply
Tensor.__neg__ = negative
Tensor.__pow__ = power
Tensor.__rpow__ = rpower
Tensor.prod = prod
Tensor.__sub__ = subtract
Tensor.__rsub__ = rsubtract
Tensor.sum = sum


__all__ = [
    "add",
    "divide",
    "exp",
    "log",
    "matmul",
    "mean",
    "multiply",
    "power",
    "prod",
    "sqrt",
    "square",
    "subtract",
    "sum"
]

from prml.function.math.add import add
from prml.function.math.divide import divide, rdivide
from prml.function.math.exp import exp
from prml.function.math.log import log
from prml.function.math.matmul import matmul, rmatmul
from prml.function.math.mean_squared_error import mean_squared_error
from prml.function.math.mean import mean
from prml.function.math.multiply import multiply
from prml.function.math.negative import negative
from prml.function.math.power import power, rpower
from prml.function.math.sqrt import sqrt
from prml.function.math.square import square
from prml.function.math.subtract import subtract, rsubtract
from prml.function.math.sum_squared_error import sum_squared_error
from prml.function.math.sum import sum


from prml.tensor.tensor import Tensor
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
Tensor.__sub__ = subtract
Tensor.__rsub__ = rsubtract
Tensor.sum = sum


__all__ = [
    "exp",
    "log",
    "mean_squared_error",
    "mean",
    "sqrt",
    "square",
    "sum_squared_error",
    "sum"
]

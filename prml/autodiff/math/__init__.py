from prml.autodiff.math.add import add
from prml.autodiff.math.divide import divide, rdivide
from prml.autodiff.math.matmul import matmul, rmatmul
from prml.autodiff.math.mean import mean
from prml.autodiff.math.multiply import multiply
from prml.autodiff.math.negative import negative
from prml.autodiff.math.power import power, rpower
from prml.autodiff.math.subtract import subtract, rsubtract
from prml.autodiff.math.sum import sum
from prml.autodiff.core import Array


Array.__add__ = add
Array.__radd__ = add
Array.__truediv__ = divide
Array.__rtruediv__ = rdivide
Array.__matmul__ = matmul
Array.__rmatmul__ = rmatmul
Array.mean = mean
Array.__mul__ = multiply
Array.__rmul__ = multiply
Array.__neg__ = negative
Array.__pow__ = power
Array.__rpow__ = rpower
Array.__sub__ = subtract
Array.__rsub__ = rsubtract
Array.sum = sum

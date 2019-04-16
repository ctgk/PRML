from prml.nn.math.negative import negative
from prml.nn.math.add import add
from prml.nn.math.subtract import subtract, rsubtract
from prml.nn.math.divide import divide, rdivide
from prml.nn.math.mean import mean
from prml.nn.math.multiply import multiply
from prml.nn.math.matmul import matmul, rmatmul
from prml.nn.math.power import power, rpower
from prml.nn.math.sum import sum
from prml.nn.array import Array


Array.__neg__ = negative
Array.__add__ = add
Array.__radd__ = add
Array.__sub__ = subtract
Array.__rsub__ = rsubtract
Array.__truediv__ = divide
Array.__rtruediv__ = rdivide
Array.__mul__ = multiply
Array.__rmul__ = multiply
Array.__matmul__ = matmul
Array.__rmatmul__ = rmatmul
Array.__pow__ = power
Array.__rpow__ = rpower
Array.sum = sum
Array.mean = mean

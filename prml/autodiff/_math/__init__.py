from prml.autodiff._math._add import add
from prml.autodiff._math._divide import divide, _rdivide
from prml.autodiff._math._mean import mean
from prml.autodiff._math._multiply import multiply
from prml.autodiff._math._negative import negative
from prml.autodiff._math._power import power, rpower
from prml.autodiff._math._subtract import subtract, rsubtract
from prml.autodiff._math._sum import sum
from prml.autodiff._core import Array


Array.__add__ = add
Array.__radd__ = add
Array.__truediv__ = divide
Array.__rtruediv__ = _rdivide
Array.mean = mean
Array.__mul__ = multiply
Array.__rmul__ = multiply
Array.__neg__ = negative
Array.__pow__ = power
Array.__rpow__ = rpower
Array.__sub__ = subtract
Array.__rsub__ = rsubtract
Array.sum = sum

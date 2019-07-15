from prml.autodiff._core._array import Array
from prml.autodiff._core._backprop import backprop
from prml.autodiff._core._reshape import _reshape_method


Array.backprop = backprop
Array.reshape = _reshape_method

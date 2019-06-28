from prml.autodiff._core._array import Array
from prml.autodiff._core._backprop import backprop


Array.backprop = backprop

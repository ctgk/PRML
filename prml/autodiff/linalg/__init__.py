from prml.autodiff.linalg._cholesky import cholesky
from prml.autodiff.linalg._determinant import det
from prml.autodiff.linalg._inverse import inv
from prml.autodiff.linalg._log_determinant import logdet
from prml.autodiff.linalg._matmul import matmul, rmatmul
from prml.autodiff.linalg._solve import solve
from prml.autodiff.linalg._trace import trace
from prml.autodiff._core._array import Array


Array.__matmul__ = matmul
Array.__rmatmul__ = rmatmul
Array.inv = inv


__all__ = [
    "cholesky",
    "det",
    "inv",
    "logdet",
    "matmul",
    "solve",
    "trace"
]

import numpy as np
from prml.autodiff._core._function import _Function, broadcast_to


class _ChiSquare(_Function):

    def _forward(self, df):
        self.output = np.random.chisquare(df)
        return self.output

    def _backward(self, delta, df):
        ddf = self.output / df
        return ddf


def chi_square(df, size: tuple = None):
    if size is not None:
        df = broadcast_to(df, size)
    return _ChiSquare().forward(df)

from prml.autodiff._core._function import broadcast_to
from prml.autodiff._math._sqrt import sqrt
from prml.autodiff.random._chi_square import chi_square
from prml.autodiff.random._gaussian import gaussian


def students_t(mean, precision, df, size=None):
    if size is not None:
        mean = broadcast_to(mean, size)
        precision = broadcast_to(precision, size)
        df = broadcast_to(df, size)
    n = gaussian(0, 1 / sqrt(precision))
    x2 = chi_square(df)
    sample = mean + n * sqrt(df / x2)
    return sample

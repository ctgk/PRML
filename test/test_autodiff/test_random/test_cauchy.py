import unittest

import numpy as np

from prml import autodiff


class TestCauchy(unittest.TestCase):

    def test_cauchy(self):
        pass

    def test_cauchy_logpdf(self):
        autodiff.config.dtype = np.float64
        x = autodiff.asarray(1)
        loc = autodiff.asarray(0.5)
        scale = autodiff.asarray(0.6)
        expect_dx, expect_dloc, expect_dscale = autodiff.numerical_gradient(
            autodiff.random.cauchy_logpdf, 1e-6, x, loc, scale)
        autodiff.random.cauchy_logpdf(x, loc, scale).backprop()
        actual_dx, actual_dloc, actual_dscale = (
            x.grad[0], loc.grad[0], scale.grad[0])
        self.assertAlmostEqual(expect_dx.value[0], actual_dx)
        self.assertAlmostEqual(expect_dloc.value[0], actual_dloc)
        self.assertAlmostEqual(expect_dscale.value[0], actual_dscale)


if __name__ == "__main__":
    unittest.main()

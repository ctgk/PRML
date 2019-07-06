import unittest

import numpy as np

from prml import autodiff


class TestLogdet(unittest.TestCase):

    def setUp(self):
        self.dtype_previous = autodiff.config.dtype
        autodiff.config.dtype = np.float64

    def tearDown(self):
        autodiff.config.dtype = self.dtype_previous

    def test_logdet(self):
        x = np.random.randn(7, 7)
        x = x @ x.T
        expect = np.linalg.slogdet(x)[1]
        actual = autodiff.linalg.logdet(x).value
        self.assertTrue(np.allclose(expect, actual), msg=f"{expect}\n{actual}")

        x = autodiff.asarray(x)
        eps = np.zeros((7, 7), dtype=np.float64)
        eps[2, 2] = 1e-8
        expect_dx, = autodiff.numerical_gradient(
            autodiff.linalg.logdet, eps, x
        )
        autodiff.linalg.logdet(x).backprop()
        actual_dx = x.grad[2, 2]
        self.assertAlmostEqual(expect_dx.value[0], actual_dx, places=3)


if __name__ == "__main__":
    unittest.main()

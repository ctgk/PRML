import unittest

import numpy as np

from prml import autodiff


class TestInv(unittest.TestCase):

    def setUp(self):
        self.dtype_previous = autodiff.config.dtype
        autodiff.config.dtype = np.float64

    def tearDown(self):
        autodiff.config.dtype = self.dtype_previous

    def test_inv(self):
        x = np.random.randn(3, 3)
        expect = np.linalg.inv(x)
        actual = autodiff.linalg.inv(x).value
        self.assertTrue(np.allclose(expect, actual))

        x = autodiff.asarray(x)
        eps = np.zeros((3, 3), dtype=np.float64)
        eps[0, 0] = 1e-8
        expect_dx, = autodiff.numerical_gradient(
            lambda a: autodiff.linalg.inv(a).sum(), eps, x
        )
        autodiff.linalg.inv(x).sum().backprop()
        actual_dx = x.grad[0, 0]
        self.assertAlmostEqual(expect_dx.value[0], actual_dx, places=3)


if __name__ == "__main__":
    unittest.main()

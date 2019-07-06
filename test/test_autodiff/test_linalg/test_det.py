import unittest

import numpy as np

from prml import autodiff


class TestDet(unittest.TestCase):

    def setUp(self):
        autodiff.config.dtype = np.float64

    def tearDown(self):
        autodiff.config.dtype = np.float32

    def test_det(self):
        x = np.random.randn(6, 6)
        expect = np.linalg.det(x)
        actual = autodiff.linalg.det(x).value
        self.assertTrue(np.allclose(expect, actual))

        x = autodiff.asarray(x)
        eps = np.zeros((6, 6), dtype=np.float64)
        eps[1, 1] = 1e-8
        expect_dx, = autodiff.numerical_gradient(
            lambda a: autodiff.linalg.det(a), eps, x
        )
        autodiff.linalg.det(x).backprop()
        actual_dx = x.grad[1, 1]
        self.assertAlmostEqual(expect_dx.value[0], actual_dx, places=5)


if __name__ == "__main__":
    unittest.main()

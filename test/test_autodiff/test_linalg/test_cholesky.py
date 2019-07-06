import unittest

import numpy as np

from prml import autodiff


class TestCholesky(unittest.TestCase):

    def setUp(self):
        autodiff.config.dtype = np.float64

    def tearDown(self):
        autodiff.config.dtype = np.float32

    def test_cholesky(self):
        npx = np.random.randn(5, 5)
        npx = npx @ npx.T
        expect = np.linalg.cholesky(npx)
        actual = autodiff.linalg.cholesky(npx).value
        self.assertTrue(np.allclose(expect, actual))

        x = autodiff.asarray(npx)
        eps = np.zeros((5, 5))
        eps[1, 1] = 1e-8
        expect_dx, = autodiff.numerical_gradient(
            lambda x_: autodiff.linalg.cholesky(x_).sum(), eps, x
        )
        autodiff.linalg.cholesky(x).sum().backprop()
        actual_dx = x.grad[1, 1]
        self.assertAlmostEqual(expect_dx.value[0], actual_dx, places=5)


if __name__ == "__main__":
    unittest.main()

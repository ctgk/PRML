import unittest
import numpy as np
from prml import autodiff


class TestMatmul(unittest.TestCase):

    def setUp(self):
        self.dtype_previous = autodiff.config.dtype
        autodiff.config.dtype = np.float64

    def tearDown(self):
        autodiff.config.dtype = self.dtype_previous

    def test_matmul(self):
        npa = np.random.randn(3, 2, 4, 6)
        npb = np.random.randn(2, 6, 3)
        a = autodiff.asarray(npa)
        b = autodiff.asarray(npb)
        c = a @ b
        self.assertTrue(np.allclose(c.value, npa @ npb))

        npg = np.random.randn(3, 2, 4, 3)
        c.backprop(npg)
        self.assertTrue(np.allclose(a.grad, npg @ np.swapaxes(npb, -1, -2)))
        self.assertTrue(
            np.allclose(b.grad, (np.swapaxes(npa, -1, -2) @ npg).sum(axis=0)))


if __name__ == "__main__":
    unittest.main()

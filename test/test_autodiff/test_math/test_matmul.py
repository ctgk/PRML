import unittest
import numpy as np
from prml import autodiff


class TestMatmul(unittest.TestCase):

    def test_matmul(self):
        npa = np.random.randn(4, 6)
        npb = np.random.randn(6, 3)
        a = autodiff.asarray(npa)
        b = autodiff.asarray(npb)
        c = a @ b
        self.assertTrue(np.allclose(c.value, npa @ npb))

        npg = np.random.randn(4, 3)
        c.backward(npg)
        self.assertTrue(np.allclose(a.grad, npg @ npb.T))
        self.assertTrue(np.allclose(b.grad, npa.T @ npg))


if __name__ == "__main__":
    unittest.main()

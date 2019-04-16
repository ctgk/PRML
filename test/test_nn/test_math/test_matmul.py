import unittest
import numpy as np
import prml.nn as nn


class TestMatmul(unittest.TestCase):

    def test_matmul(self):
        npa = np.random.randn(4, 6)
        npb = np.random.randn(6, 3)
        a = nn.asarray(npa)
        b = nn.asarray(npb)
        c = a @ b
        self.assertTrue(np.allclose(c.value, npa @ npb))

        npg = np.random.randn(4, 3)
        c.backward(npg)
        self.assertTrue(np.allclose(a.grad, npg @ npb.T))
        self.assertTrue(np.allclose(b.grad, npa.T @ npg))


if __name__ == "__main__":
    unittest.main()

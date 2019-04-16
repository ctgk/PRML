import unittest
import numpy as np
import prml.nn as nn


class TestAdd(unittest.TestCase):

    def test_add(self):
        npa = np.random.randn(4, 5)
        npb = np.random.randn(4, 5)
        a = nn.asarray(npa)
        b = nn.asarray(npb)
        c = a + b
        self.assertTrue(np.allclose(c.value, npa + npb))

        npg = np.random.randn(4, 5)
        c.backward(npg)
        self.assertTrue(np.allclose(a.grad, npg))
        self.assertTrue(np.allclose(b.grad, npg))

    def test_add_bias(self):
        npa = np.random.randn(4, 3)
        npb = np.random.randn(3)
        a = nn.asarray(npa)
        b = nn.asarray(npb)
        c = a + b
        self.assertTrue(np.allclose(c.value, npa + npb))

        npg = np.random.randn(4, 3)
        c.backward(npg)
        self.assertTrue(np.allclose(a.grad, npg))
        self.assertTrue(np.allclose(b.grad, npg.sum(axis=0)))

    def test_add_scalar(self):
        npa = np.random.randn(5, 6)
        npb = 2
        a = nn.asarray(npa)
        b = nn.asarray(npb)
        c = a + b
        self.assertTrue(np.allclose(c.value, npa + npb))

        npg = np.random.randn(5, 6)
        c.backward(npg)
        self.assertTrue(np.allclose(a.grad, npg))
        self.assertTrue(np.allclose(b.grad, np.sum(npg)))


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import prml.nn as nn


class TestMultiply(unittest.TestCase):

    def test_multiply(self):
        npx = np.random.randn(5, 6)
        npy = np.random.randn(5, 6)
        x = nn.asarray(npx)
        y = nn.asarray(npy)
        z = x * y
        self.assertTrue(np.allclose(z.value, npx * npy))

        npg = np.random.randn(5, 6)
        z.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg * npy))
        self.assertTrue(np.allclose(y.grad, npg * npx))

    def test_multiply_vector(self):
        npx = np.random.randn(5, 6)
        npy = np.random.randn(6)
        x = nn.asarray(npx)
        y = nn.asarray(npy)
        z = x * y
        self.assertTrue(np.allclose(z.value, npx * npy))

        npg = np.random.randn(5, 6)
        z.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg * npy))
        self.assertTrue(np.allclose(y.grad, np.sum(npg * npx, axis=0)))

    def test_multiply_scalar(self):
        npx = np.random.randn(5, 6)
        npy = 3.5
        x = nn.asarray(npx)
        y = nn.asarray(npy)
        z = x * y
        self.assertTrue(np.allclose(z.value, npx * npy))

        npg = np.random.randn(5, 6)
        z.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg * npy))
        self.assertTrue(np.allclose(y.grad, np.sum(npg * npx)))


if __name__ == "__main__":
    unittest.main()


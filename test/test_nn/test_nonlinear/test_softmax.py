import unittest
import numpy as np
import prml.nn as nn


class TestSoftmax(unittest.TestCase):

    def test_forward(self):
        npx = np.random.randn(5, 3)
        npy = np.exp(npx) / np.exp(npx).sum(axis=-1, keepdims=True)
        self.assertTrue(np.allclose(npy, nn.softmax(npx).value))

    def test_backward(self):
        npx = np.random.randn(1, 4)
        x = nn.asarray(npx)
        y = nn.square(nn.softmax(x)).sum()
        y.backward()
        grad = x.grad

        eps = np.zeros(4)
        eps[0] = 1e-3
        numerical_grad = (nn.square(nn.softmax(npx + eps)).sum() - nn.square(nn.softmax(npx - eps)).sum()) / 2e-3
        self.assertAlmostEqual(grad[0][0], numerical_grad.value[0], 3)


if __name__ == "__main__":
    unittest.main()

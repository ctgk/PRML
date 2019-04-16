import unittest
import numpy as np
import prml.nn as nn


class TestSigmoidCrossEntropy(unittest.TestCase):

    def test_sigmoid_cross_entropy(self):
        npx = np.random.randn(10, 3)
        npy = np.tanh(npx * 0.5) * 0.5 + 0.5
        npt = np.random.uniform(0, 1, (10, 3))
        x = nn.asarray(npx)
        t = nn.asarray(npt)
        loss = nn.loss.sigmoid_cross_entropy(x, t)
        self.assertTrue(np.allclose(loss.value, -npt * np.log(npy) - (1 - npt) * np.log(1 - npy)))

        npg = np.random.randn(10, 3)
        loss.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg * (npy - npt)))
        self.assertTrue(np.allclose(t.grad, -npg * npx))


if __name__ == "__main__":
    unittest.main()

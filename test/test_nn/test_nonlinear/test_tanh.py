import unittest
import numpy as np
import prml.nn as nn


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        npx = np.random.randn(4, 7)
        x = nn.asarray(npx)
        y = nn.tanh(x)
        self.assertTrue(np.allclose(y.value, np.tanh(npx)))

        npg = np.random.randn(4, 7)
        y.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg * (1 - y.value ** 2)))


if __name__ == "__main__":
    unittest.main()

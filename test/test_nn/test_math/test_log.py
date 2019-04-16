import unittest
import numpy as np
import prml.nn as nn


class TestLog(unittest.TestCase):

    def test_log(self):
        npx = np.random.uniform(0, 10, (4, 5))
        x = nn.asarray(npx)
        y = nn.log(x)
        self.assertTrue(np.allclose(y.value, np.log(npx)))

        npg = np.random.randn(4, 5)
        y.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg / npx))


if __name__ == "__main__":
    unittest.main()

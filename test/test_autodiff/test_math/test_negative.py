import unittest
import numpy as np
from prml import autodiff


class TestNegative(unittest.TestCase):

    def test_negative(self):
        npx = np.random.randn(8, 9)
        x = autodiff.asarray(npx)
        y = -x
        self.assertTrue(np.allclose(y.value, -npx))

        npg = np.random.randn(8, 9)
        y.backprop(npg)
        self.assertTrue(np.allclose(x.grad, -npg))


if __name__ == "__main__":
    unittest.main()

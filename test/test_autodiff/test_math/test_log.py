import unittest
import numpy as np
from prml import autodiff


class TestLog(unittest.TestCase):

    def test_log(self):
        npx = np.random.uniform(0, 10, (4, 5))
        x = autodiff.asarray(npx)
        y = autodiff.log(x)
        self.assertTrue(np.allclose(y.value, np.log(npx)))

        npg = np.random.randn(4, 5)
        y.backprop(npg)
        self.assertTrue(np.allclose(x.grad, npg / npx))


if __name__ == "__main__":
    unittest.main()

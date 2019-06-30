import unittest

import numpy as np

from prml import autodiff


class TestLogit(unittest.TestCase):

    def test_logit(self):
        npx = np.random.uniform(0, 1, (3, 5))
        x = autodiff.asarray(npx)
        y = autodiff.logit(x)
        self.assertTrue(np.allclose(y.value, np.log(npx / (1 - npx))))

        y.backprop()
        self.assertTrue(np.allclose(x.grad, 1 / x.value / (1 - x.value)))


if __name__ == "__main__":
    unittest.main()

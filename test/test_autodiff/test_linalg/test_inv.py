import unittest

import numpy as np

from prml import autodiff


class TestInv(unittest.TestCase):

    def test_inv(self):
        x = np.random.randn(3, 3)
        expect = np.linalg.inv(x)
        actual = autodiff.linalg.inv(x).value
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()

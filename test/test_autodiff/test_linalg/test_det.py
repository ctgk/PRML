import unittest

import numpy as np

from prml import autodiff


class TestDet(unittest.TestCase):

    def test_det(self):
        x = np.random.randn(6, 6)
        expect = np.linalg.det(x)
        actual = autodiff.linalg.det(x).value
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from prml import autodiff


class TestTrace(unittest.TestCase):

    def test_trace(self):
        x = np.random.randn(2, 2)
        expect = np.trace(x)
        actual = autodiff.linalg.trace(x).value
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()

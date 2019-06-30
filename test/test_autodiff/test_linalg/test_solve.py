import unittest

import numpy as np

from prml import autodiff


class TestSolve(unittest.TestCase):

    def test_solve(self):
        A = np.random.randn(4, 4)
        b = np.random.randn(4, 1)
        expect = np.linalg.solve(A, b)
        actual = autodiff.linalg.solve(A, b).value
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()

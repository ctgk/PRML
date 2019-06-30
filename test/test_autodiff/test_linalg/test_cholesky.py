import unittest

import numpy as np

from prml import autodiff


class TestCholesky(unittest.TestCase):

    def test_cholesky(self):
        npx = np.random.randn(5, 5)
        npx = npx @ npx.T
        expect = np.linalg.cholesky(npx)
        actual = autodiff.linalg.cholesky(npx).value
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()

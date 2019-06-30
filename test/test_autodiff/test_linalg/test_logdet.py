import unittest

import numpy as np

from prml import autodiff


class TestLogdet(unittest.TestCase):

    def test_logdet(self):
        x = np.random.randn(7, 7)
        x = x @ x.T
        expect = np.linalg.slogdet(x)[1]
        actual = autodiff.linalg.logdet(x).value
        self.assertTrue(np.allclose(expect, actual), msg=f"{expect}\n{actual}")


if __name__ == "__main__":
    unittest.main()

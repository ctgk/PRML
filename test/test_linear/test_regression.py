import unittest

import numpy as np

from prml import linear


class TestRegression(unittest.TestCase):

    def test_fit(self):
        model = linear.Regression()
        model.fit(X=np.arange(10).reshape(10, 1), t=np.arange(10))
        self.assertTrue(np.allclose(np.ones((1,)), model.w))

    def test_predict(self):
        model = linear.Regression()
        model.w = np.ones((1,)) * 2
        actual = model.predict(np.arange(10, 20, 1).reshape(10, 1))
        expect = np.arange(20, 40, 2).reshape(10)
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()

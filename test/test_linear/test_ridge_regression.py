import unittest

import numpy as np

from prml.linear import RidgeRegression


class TestRidgeRegression(unittest.TestCase):

    def test_fit_predict(self):
        x_train = np.array([-1, 0, 1]).reshape(-1, 1)
        y_train = np.array([-2, 0, 2])
        model = RidgeRegression(alpha=1.)
        model.fit(x_train, y_train)
        self.assertTrue(np.allclose(model.w, np.array([4 / 3])))

        actual = model.predict(np.array([[3]]))
        self.assertTrue(np.allclose(actual, np.array([4])))


if __name__ == '__main__':
    unittest.main()

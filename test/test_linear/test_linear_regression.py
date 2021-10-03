import unittest

import numpy as np

from prml.linear import LinearRegression


class TestLinearRegression(unittest.TestCase):

    def test_fit(self):
        x_train = np.array([-1, 0, 1]).reshape(-1, 1)
        y_train = np.array([-2, 0, 2])
        model = LinearRegression()

        model.fit(x_train, y_train)

        self.assertTrue(
            np.allclose(model.w, np.array([2])),
        )

    def test_predict(self):
        x_train = np.array([-1, 0, 1]).reshape(-1, 1)
        y_train = np.array([-2, 0, 2])
        model = LinearRegression()
        model.fit(x_train, y_train)

        actual = model.predict(np.array([[3]]))
        self.assertTrue(np.allclose(actual, np.array([6])))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from prml.linear_regressors import LinearRegressor


class TestLinearRegressor(unittest.TestCase):

    def test_regression(self):
        regressor = LinearRegressor()
        X = np.array([[0.], [1.]])
        y = np.array([0., 1.])
        regressor.fit(X, y)
        self.assertEqual(regressor.w, 1)

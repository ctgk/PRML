import unittest
import numpy as np
from prml.linear_regressors import RidgeRegressor


class TestRidgeRegressor(unittest.TestCase):

    def test_fit(self):
        regressor = RidgeRegressor(alpha=0.)
        X = np.array([[0.], [1.]])
        y = np.array([0., 1.])
        regressor.fit(X, y)
        self.assertEqual(regressor.w, 1)

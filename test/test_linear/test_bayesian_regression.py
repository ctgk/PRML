import unittest

import numpy as np

from prml.linear import BayesianRegression


class TestBayesianRegression(unittest.TestCase):

    def test_fit_predict(self):
        x_train = np.array([-1, 0, 1]).reshape(-1, 1)
        y_train = np.array([-2, 0, 2])
        model = BayesianRegression(alpha=1., beta=1.)
        model.fit(x_train, y_train)
        self.assertTrue(np.allclose(model.w_mean, np.array([4 / 3])))

        mean, std = model.predict(np.array([[3]]), return_std=True)
        self.assertTrue(np.allclose(mean, np.array([4])))
        self.assertTrue(np.allclose(std, np.array([2])))


if __name__ == '__main__':
    unittest.main()

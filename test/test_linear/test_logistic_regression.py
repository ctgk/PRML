import unittest

import numpy as np

from prml.linear import LogisticRegression


class TestLogisticRegression(unittest.TestCase):

    def test_fit_classify_proba(self):
        x_train = np.array([-3, -2, -1, 1, 2, 3]).reshape(-1, 1)
        y_train = np.array([0, 0, 1, 0, 1, 1])
        model = LogisticRegression()
        model.fit(x_train, y_train)
        self.assertTrue(np.allclose(model.w, np.array([0.73248753])))

        actual = model.classify(np.array([[-5], [5]]))
        self.assertTrue(np.allclose(actual, np.array([0, 1])))

        actual = model.proba(np.array([[0], [4]]))
        self.assertTrue(np.allclose(actual, np.array([0.5, 0.94930727])))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from prml.linear_classifiers import LogisticRegressor


class TestLogisticRegressor(unittest.TestCase):

    def test_predict_proba(self):
        classifier = LogisticRegressor()
        X = np.array([[-1.], [1.]])
        y = np.array([0., 1.])
        classifier.fit(X, y)
        self.assertEqual(classifier.predict_proba(np.zeros((1, 1))), 0.5)

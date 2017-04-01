import unittest
import numpy as np
from prml.features import PolynomialFeatures


class TestPolynomialFeatures(unittest.TestCase):

    def test_transform(self):
        self.features = PolynomialFeatures()
        x = np.array([[0., 1.], [2., 3.]])
        X = np.array([
            [1., 0., 1., 0., 0., 1.],
            [1., 2., 3., 4., 6., 9.]])
        self.assertTrue(np.allclose(self.features.transform(x), X))

import unittest
import numpy as np
import prml.autograd as ag


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):
        self.assertRaises(ValueError, ag.random.Gaussian, 0, -1)
        self.assertRaises(ValueError, ag.random.Gaussian, 0, np.array([1, -1]))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import prml.neural_networks as pn


class TestSigmoidCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.obj = pn.losses.SigmoidCrossEntropy()

    def test_call(self):
        x = np.array([100, -100, -100], np.float32).reshape(-1, 1)
        t = np.array([1., 0., 0.], np.float32).reshape(-1, 1)
        self.assertAlmostEqual(
            self.obj(x, t), 0)

    def test_multilabel(self):
        x = np.array([100, 100, -100], np.float32).reshape(1, -1)
        t = np.array([1., 1., 0.], np.float32).reshape(1, -1)
        self.assertAlmostEqual(
            self.obj(x, t), 0)

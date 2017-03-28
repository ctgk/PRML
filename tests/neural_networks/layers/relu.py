import unittest
import numpy as np
import prml.neural_networks as pn


class TestReLU(unittest.TestCase):

    def setUp(self):
        self.obj = pn.layers.ReLU()

    def test_forward(self):
        x = np.array([-10, 0, 10], np.float32)
        expected = np.array([0, 0, 10], np.float)
        actual = self.obj.forward(x)
        self.assertTrue(
            np.allclose(expected, actual))

    def test_backward(self):
        x = np.array([-10, 0, 10], dtype=np.float32)
        self.obj.forward(x, True)
        expected = np.array([0, 0, 1], dtype=np.float32)
        actual = self.obj.backward(np.ones_like(x))
        self.assertTrue(
            np.allclose(expected, actual))

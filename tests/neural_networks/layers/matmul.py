import unittest
import numpy as np
import prml.neural_networks as pn


class TestMatMul(unittest.TestCase):

    def test_forward(self):
        self.obj = pn.layers.MatMul(5, 4)
        x = np.random.standard_normal((10, 5)).astype(np.float32)
        expected = x @ self.obj.param
        self.assertTrue(np.allclose(self.obj.forward(x), expected))

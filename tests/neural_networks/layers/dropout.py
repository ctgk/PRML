import unittest
import numpy as np
import prml.neural_networks as pn


class TestDropout(unittest.TestCase):

    def test_training(self):
        dropout = pn.layers.Dropout(0.999)
        x = np.random.randint(1, 10, 10).astype(np.float32)
        y = dropout.forward(x, training=True)
        z = dropout.backward(y)
        self.assertTrue(np.any(y == 0))
        self.assertTrue(np.allclose(z, y * 1000.))

    def test_evaluation(self):
        dropout = pn.layers.Dropout(0.999)
        x = np.random.randint(1, 10, 10).astype(np.float32)
        y = dropout.forward(x)
        self.assertTrue(
            np.allclose(x, y))

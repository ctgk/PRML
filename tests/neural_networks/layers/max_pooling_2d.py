import unittest
import numpy as np
import prml.neural_networks as pn


class TestMaxPooling2d(unittest.TestCase):

    def test_shape(self):
        pooling = pn.layers.MaxPooling2d(2, 2)
        A = np.random.randint(0, 10, (50, 28, 28, 3)).astype(np.float32)
        B = pooling.forward(A, True)
        self.assertEqual(B.shape, (50, 14, 14, 3))
        C = pooling.backward(B)
        self.assertEqual(C.shape, (50, 28, 28, 3))

        pooling = pn.layers.MaxPooling2d(5, 2, pad=1)
        B = pooling.forward(A, True)
        self.assertEqual(B.shape, (50, 13, 13, 3))
        C = pooling.backward(B)
        self.assertEqual(C.shape, (50, 28, 28, 3))

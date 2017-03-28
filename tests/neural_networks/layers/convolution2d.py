import unittest
import numpy as np
import prml.neural_networks as pn


class TestConvolution2d(unittest.TestCase):

    def test_shape(self):
        conv = pn.layers.Convolution2d(3, 32, 3, pad=1)
        A = np.random.rand(50, 28, 28, 3).astype(np.float32)
        B = conv.forward(A, True)
        self.assertEqual(B.shape, (50, 28, 28, 32))
        C = conv.backward(B)
        self.assertEqual(C.shape, A.shape)

        conv = pn.layers.Convolution2d(1, 3, 5, stride=2, pad=0)
        A = np.random.rand(50, 28, 28, 1).astype(np.float32)
        B = conv.forward(A, True)
        self.assertEqual(B.shape, (50, 12, 12, 3))
        C = conv.backward(B)
        self.assertEqual(C.shape, A.shape)

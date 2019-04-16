import unittest
import numpy as np
from scipy.ndimage.filters import correlate
import prml.nn as nn


class TestConvolve2d(unittest.TestCase):

    def test_convolve2d_forward(self):
        img = np.random.randn(1, 5, 5, 1)
        kernel = np.random.randn(3, 3, 1, 1)
        output = nn.convolve2d(img, kernel)
        self.assertTrue(
            np.allclose(
                output.value[0, ..., 0],
                correlate(img[0, ..., 0], kernel[..., 0, 0])[1:-1, 1:-1]
            )
        )
        self.assertEqual(nn.config.dtype, np.float32)
        self.assertEqual(output.value.dtype, nn.config.dtype)

    def test_convolve2d_backward(self):
        x = nn.random.normal(0, 1, (1, 5, 5, 1))
        w = nn.random.normal(0, 1, (3, 3, 1, 1))
        for _ in range(1000):
            x.cleargrad()
            w.cleargrad()
            output = nn.convolve2d(x, w, (2, 2), (1, 1))
            output.backward(2 * (output.value - 1))
            x.value -= x.grad * 0.01
            w.value -= w.grad * 0.01
        self.assertTrue(np.allclose(output.value, 1))
        self.assertEqual(nn.config.dtype, np.float32)
        self.assertEqual(x.dtype, nn.config.dtype)
        self.assertEqual(w.dtype, nn.config.dtype)
        self.assertEqual(output.dtype, nn.config.dtype)

    def test_convolve2d_network(self):
        x = nn.random.normal(0, 1, (1, 5, 5, 1))
        kernel = nn.random.normal(0, 1, (3, 3, 1, 1))
        conv = nn.image.Convolve2d(kernel, (1, 1), (0, 0))
        for _ in range(1000):
            x.cleargrad()
            conv.clear()
            output = conv(x)
            output.backward(2 * (output.value - 1))
            x.value -= x.grad * 0.01
            for param in conv.parameter.values():
                param.value -= param.grad * 0.01
        self.assertTrue(np.allclose(output.value, 1))


if __name__ == "__main__":
    unittest.main()

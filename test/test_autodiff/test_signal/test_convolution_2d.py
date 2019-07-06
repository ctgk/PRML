import unittest

import numpy as np
from scipy.ndimage.filters import correlate

from prml import autodiff


class TestConvolution2d(unittest.TestCase):

    def test_convolution_2d_forward(self):
        img = np.random.randn(1, 5, 5, 1).astype(np.float32)
        kernel = np.random.randn(3, 3, 1, 1).astype(np.float32)
        output = autodiff.signal.convolution_2d(img, kernel)
        self.assertTrue(
            np.allclose(
                output.value[0, ..., 0],
                correlate(img[0, ..., 0], kernel[..., 0, 0])[1:-1, 1:-1]
            )
        )
        self.assertEqual(output.value.dtype, autodiff.config.dtype)

    def test_convolution_2d_backward(self):
        x = autodiff.random.gaussian(0, 1, (1, 5, 5, 1))
        w = autodiff.random.gaussian(0, 1, (3, 3, 1, 1))
        for _ in range(1000):
            x.cleargrad()
            w.cleargrad()
            output = autodiff.signal.convolution_2d(x, w, (2, 2), (1, 1))
            output.backprop(2 * (output.value - 1))
            x.value -= x.grad * 0.01
            w.value -= w.grad * 0.01
        self.assertTrue(np.allclose(output.value, 1))
        self.assertEqual(autodiff.config.dtype, np.float32)
        self.assertEqual(x.dtype, autodiff.config.dtype)
        self.assertEqual(w.dtype, autodiff.config.dtype)
        self.assertEqual(output.dtype, autodiff.config.dtype)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from scipy.ndimage.filters import correlate

import prml.nn as nn


class TestDeconvolve2d(unittest.TestCase):

    def test_deconvolve2d_forward(self):
        img = np.random.randn(1, 3, 3, 1).astype(np.float32)
        kernel = np.random.randn(3, 3, 1, 1).astype(np.float32)
        out = nn.deconvolve2d(img, kernel, (1, 1), (0, 0))

        actual = out.value[0, 1:-1, 1:-1, 0]
        expect = correlate(
            img[0, :, :, 0],
            kernel[::-1, ::-1, 0, 0],
            mode='constant',
        )
        self.assertTrue(np.allclose(actual, expect, 0, 1e-2))

    def test_deconvolve2d_backward(self):
        x = nn.random.normal(0, 1, (1, 3, 3, 1))
        w = nn.random.normal(0, 1, (3, 3, 1, 1))
        for _ in range(1000):
            x.cleargrad()
            w.cleargrad()
            output = nn.deconvolve2d(x, w, (2, 2), (1, 1))
            output.backward(2 * (output.value - 1))
            x.value -= x.grad * 0.01
            w.value -= w.grad * 0.01
        self.assertTrue(
            np.allclose(output.value, 1, rtol=0, atol=1e-2),
            output.value,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from prml import autodiff


class TestMaxPooling2d(unittest.TestCase):

    def test_max_pooling2d_forward(self):
        img = np.array([
            [2, 3, 4, 1],
            [2, 5, 1, 2],
            [3, 5, 1, 3],
            [3, 7, 8, 2]
        ]).astype(np.float32)
        img = img[None, :, :, None]
        expected = np.array([[5, 4], [7, 8]])
        actual = autodiff.signal.max_pooling_2d(img, 2, 2).value.squeeze()
        self.assertTrue((expected == actual).all(), actual)

    def test_max_pooling2d_backward(self):
        img = np.array([
            [2, 3, 4, 1],
            [2, 5, 1, 2],
            [3, 5, 1, 3],
            [3, 7, 8, 2]
        ]).astype(np.float32)
        img = img[None, :, :, None]
        expected = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        img = autodiff.asarray(img)
        autodiff.signal.max_pooling_2d(img, 2, 2).backprop(np.ones((1, 2, 2, 1)))
        actual = img.grad.squeeze()
        self.assertTrue((expected == actual).all())


if __name__ == "__main__":
    unittest.main()

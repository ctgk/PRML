import unittest
import numpy as np
import prml.nn as nn


class TestMaxPooling2d(unittest.TestCase):

    def test_max_pooling2d(self):
        img = np.array([
            [2, 3, 4, 1],
            [2, 5, 1, 2],
            [3, 5, 1, 3],
            [3, 7, 8, 2]
        ]).astype(np.float)
        img = img[None, :, :, None]
        expected = np.array([[5, 4], [7, 8]])
        actual = nn.max_pooling2d(img, 2, 2).value.squeeze()
        self.assertTrue((expected == actual).all(), actual)

        expected = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        img = nn.asarray(img)
        nn.max_pooling2d(img, 2, 2).backward(np.ones((1, 2, 2, 1)))
        actual = img.grad.squeeze()
        self.assertTrue((expected == actual).all())


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import prml.nn as nn


class TestLogSoftmax(unittest.TestCase):

    def test_forward(self):
        npx = np.random.randn(5, 3)
        npy = np.log(np.exp(npx) / np.exp(npx).sum(axis=-1, keepdims=True))
        self.assertTrue(np.allclose(npy, nn.log_softmax(npx).value))

    def test_backward(self):
        npx = np.random.randn(1, 5)
        x = nn.asarray(npx)
        nn.softmax(x).backward()
        grad1 = np.copy(x.grad)
        x.cleargrad()
        nn.exp(nn.log_softmax(x)).backward()
        grad2 = np.copy(x.grad)
        self.assertTrue(np.allclose(grad1, grad2))


if __name__ == "__main__":
    unittest.main()

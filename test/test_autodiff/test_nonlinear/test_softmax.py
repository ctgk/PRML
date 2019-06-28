import unittest
import numpy as np
import prml.autodiff as autodiff


class TestSoftmax(unittest.TestCase):

    def test_forward(self):
        npx = np.random.randn(5, 3)
        npy = np.exp(npx) / np.exp(npx).sum(axis=-1, keepdims=True)
        self.assertTrue(np.allclose(npy, autodiff.softmax(npx).value))

    def test_backward(self):
        npx = np.random.randn(1, 4)
        x = autodiff.asarray(npx)
        y = autodiff.square(autodiff.softmax(x)).sum()
        y.backprop()
        grad = x.grad

        eps = np.zeros(4)
        eps[0] = 1e-3
        numerical_grad = (
            autodiff.square(autodiff.softmax(npx + eps)).sum()
            - autodiff.square(autodiff.softmax(npx - eps)).sum()) / 2e-3
        self.assertAlmostEqual(grad[0][0], numerical_grad.value[0], places=3)


if __name__ == "__main__":
    unittest.main()

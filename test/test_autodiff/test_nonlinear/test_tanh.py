import unittest
import numpy as np
import prml.autodiff as autodiff


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        npx = np.random.randn(4, 7)
        x = autodiff.asarray(npx)
        y = autodiff.tanh(x)
        self.assertTrue(np.allclose(y.value, np.tanh(npx)))

        npg = np.random.randn(4, 7)
        y.backward(npg)
        self.assertTrue(np.allclose(x.grad, npg * (1 - y.value ** 2)))


if __name__ == "__main__":
    unittest.main()

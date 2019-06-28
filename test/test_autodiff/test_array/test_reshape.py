import unittest
import numpy as np

from prml.autodiff import array, reshape


class TestReshape(unittest.TestCase):

    def test_reshape(self):
        a = array(list(range(10)))
        self.assertTupleEqual((10,), a.shape)
        b = reshape(a, (2, 5))
        self.assertTupleEqual((2, 5), b.shape)
        b.backprop()
        self.assertTrue(np.allclose(np.ones((2, 5)), b.grad))
        self.assertTrue(np.allclose(np.ones(10), a.grad))


if __name__ == "__main__":
    unittest.main()

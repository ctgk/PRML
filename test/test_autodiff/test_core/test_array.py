import unittest
from unittest.mock import MagicMock
import numpy as np
from prml.autodiff.core.array import Array, array, asarray
from prml.autodiff import config


class TestArray(unittest.TestCase):

    def tearDown(self):
        config._dtype = np.float32
        config._enable_backprop = True

    def test_init(self):
        a = Array(2)
        self.assertEqual(1, a.value.ndim)
        self.assertTrue(np.allclose(2, a.value))
        self.assertIs(None, a._parent)
        self.assertIs(None, a.grad)
        self.assertIs(None, a._gradtmp)
        self.assertEqual(0, a._depth)

    def test_add_parent(self):
        mock_parent = MagicMock()
        mock_parent._out_depth.return_value = 2
        a = Array(np.ones(10))
        a.add_parent(mock_parent)
        self.assertIs(mock_parent, a._parent)
        self.assertEqual(2, a._depth)

    def test_repr(self):
        a = Array(3)
        string = repr(a)
        self.assertEqual("Array(shape=(1,), dtype=int64)", string)

    def test_ndim(self):
        a = Array(np.ones((2, 5, 8)))
        self.assertEqual(3, a.ndim)

    def test_shape(self):
        a = Array(np.zeros((7, 1, 3)))
        self.assertTupleEqual((7, 1, 3), a.shape)

    def test_size(self):
        a = Array(4.5)
        self.assertEqual(1, a.size)

    def test_cleargrad(self):
        a = Array(2.5)
        a.grad = np.array([-1.])
        a._gradtmp = np.array([6.])
        a.cleargrad()
        self.assertIs(None, a.grad)
        self.assertIs(None, a._gradtmp)

    def test_array(self):
        config.dtype = np.float16
        a = array([2, 3])
        self.assertTrue(np.allclose(np.array([2, 3]), a.value))
        self.assertEqual(config.dtype, a.dtype)

    def test_asarray(self):
        a = asarray(Array(2.7))
        self.assertTrue(np.allclose(2.7, a.value))
        b = asarray(np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10)))
        self.assertTupleEqual((2, 10, 10), b.shape)


if __name__ == "__main__":
    unittest.main()

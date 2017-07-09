import unittest
import numpy as np
from prml.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_init(self):
        t = Tensor(1)
        self.assertEqual(t.value, 1)
        self.assertIs(t.function, None)

        t = Tensor(1.)
        self.assertEqual(t.value, 1.)

        t = Tensor(np.array(0))
        self.assertEqual(t.value, 0)

        t = Tensor(np.ones(2))
        self.assertTrue((t.value == np.ones(2)).all())

        self.assertRaises(TypeError, Tensor, "abc")

    def test_repr(self):
        t = Tensor(1)
        self.assertEqual(
            repr(t),
            "Tensor(value=1)"
        )

        t = Tensor(1.)
        self.assertEqual(
            repr(t),
            "Tensor(value=1.0)"
        )

        t = Tensor(np.zeros((5, 4)))
        self.assertEqual(
            repr(t),
            "Tensor(shape=(5, 4), dtype=float64)"
        )

        t = Tensor(np.array(0))
        self.assertEqual(
            repr(t),
            "Tensor(shape=(), dtype=int64)"
        )

    def test_property(self):
        t = Tensor(1.)
        self.assertEqual(t.ndim, 0)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.size, 1)

        t = Tensor(np.ones((3, 4)))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t.size, 12)

    def test_backward(self):
        t = Tensor(1.)
        self.assertRaises(ValueError, t.backward, np.ones(1))
        self.assertRaises(TypeError, t.backward, "abc")

        t = Tensor(np.ones((2, 3)))
        self.assertRaises(ValueError, t.backward, 1)
        self.assertRaises(ValueError, t.backward, np.zeros((3, 3)))

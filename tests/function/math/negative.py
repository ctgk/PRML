import unittest
import numpy as np
from prml.tensor import Parameter


class TestNegative(unittest.TestCase):

    def test_forward_backward(self):
        x = Parameter(2.)
        y = -x
        self.assertEqual(y.value, -2)
        y.backward()
        self.assertEqual(x.grad, -1)

        x = np.random.rand(2, 3)
        xp = Parameter(x)
        y = -xp
        self.assertTrue((y.value == -x).all())
        y.backward(np.ones((2, 3)))
        self.assertTrue((xp.grad == -np.ones((2, 3))).all())

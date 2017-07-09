import unittest
import numpy as np
from prml.tensor import Parameter


class TestDivide(unittest.TestCase):

    def test_forward_backward(self):
        x = Parameter(10.)
        z = x / 2
        self.assertEqual(z.value, 5)
        z.backward()
        self.assertEqual(x.grad, 0.5)

        x = np.random.rand(5, 10, 3)
        y = np.random.rand(10, 1)
        p = Parameter(y)
        z = x / p
        self.assertTrue((z.value == x / y).all())
        z.backward(np.ones((5, 10, 3)))
        self.assertTrue((p.grad == np.sum(-x / y ** 2, axis=0).sum(axis=1, keepdims=True)).all())

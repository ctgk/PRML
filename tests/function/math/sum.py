import unittest
import numpy as np
from prml.tensor import Parameter


class TestMean(unittest.TestCase):

    def test_forward_backward(self):
        x = np.random.rand(5, 1, 2)
        xp = Parameter(x)
        z = xp.sum()
        self.assertEqual(z.value, x.sum())
        z.backward()
        self.assertTrue((xp.grad == np.ones((5, 1, 2))).all())
        xp.cleargrad()

        z = xp.sum(axis=0, keepdims=True)
        self.assertEqual(z.shape, (1, 1, 2))
        self.assertTrue((z.value == x.sum(axis=0, keepdims=True)).all())
        z.backward(np.ones((1, 1, 2)))
        self.assertTrue((xp.grad == np.ones((5, 1, 2))).all())
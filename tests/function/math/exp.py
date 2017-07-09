import unittest
import numpy as np
from prml.tensor import Parameter
from prml.function import exp


class TestExp(unittest.TestCase):

    def test_forward_backward(self):
        x = Parameter(2.)
        y = exp(x)
        self.assertEqual(y.value, np.exp(2))
        y.backward()
        self.assertEqual(x.grad, np.exp(2))

        x = np.random.rand(5, 3)
        p = Parameter(x)
        y = exp(p)
        self.assertTrue((y.value == np.exp(x)).all())
        y.backward(np.ones((5, 3)))
        self.assertTrue((p.grad == np.exp(x)).all())

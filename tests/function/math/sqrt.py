import unittest
import numpy as np
from prml.tensor import Parameter
from prml.function import sqrt


class TestSqrt(unittest.TestCase):

    def test_forward_backward(self):
        x = Parameter(2.)
        y = sqrt(x)
        self.assertEqual(y.value, np.sqrt(2))
        y.backward()
        self.assertEqual(x.grad, 0.5 / np.sqrt(2))
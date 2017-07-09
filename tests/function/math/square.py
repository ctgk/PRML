import unittest
import numpy as np
from prml.tensor import Parameter
from prml.function import square


class TestSqrt(unittest.TestCase):

    def test_forward_backward(self):
        x = Parameter(2.)
        y = square(x)
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4)
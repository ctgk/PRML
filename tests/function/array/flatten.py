import unittest
import numpy as np
from prml.tensor import Parameter
from prml.function import flatten


class TestFlatten(unittest.TestCase):

    def test_forward_backward(self):
        self.assertRaises(TypeError, flatten, "abc")
        self.assertRaises(ValueError, flatten, np.ones(1))

        x = np.random.rand(5, 4)
        p = Parameter(x)
        y = flatten(p)
        self.assertTrue((y.value == x.flatten()).all())
        y.backward(np.ones(20))
        self.assertTrue((p.grad == np.ones((5, 4))).all())

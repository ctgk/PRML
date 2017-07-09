import unittest
import numpy as np
from prml.tensor import Parameter


class TestParameter(unittest.TestCase):

    def test_init(self):
        a = Parameter(1)
        self.assertIs(a.grad, None)

    def test_backward(self):
        a = Parameter(1)
        a.backward(delta=2)
        self.assertEqual(a.grad, 2)

        a = Parameter(np.ones((3, 4)))
        a.backward(delta=np.ones((3, 4)))
        self.assertTrue((a.grad == np.ones((3, 4))).all())

    def test_cleargrad(self):
        a = Parameter(1)
        a.backward(delta=2)
        self.assertEqual(a.grad, 2)
        a.cleargrad()
        self.assertIs(a.grad, None)

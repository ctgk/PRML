import unittest
import numpy as np
import prml.autograd as ag


class TestSqrt(unittest.TestCase):

    def test_sqrt(self):
        x = ag.Parameter(2.)
        y = ag.square(x)
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4)


if __name__ == '__main__':
    unittest.main()

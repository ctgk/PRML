import unittest
import numpy as np
import prml.autograd as ag


class TestSqrt(unittest.TestCase):

    def test_sqrt(self):
        x = ag.Parameter(2.)
        y = ag.sqrt(x)
        self.assertEqual(y.value, np.sqrt(2))
        y.backward()
        self.assertEqual(x.grad, 0.5 / np.sqrt(2))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import prml.autograd as ag


class TestAdd(unittest.TestCase):

    def test_add(self):
        x = ag.Parameter(2)
        z = x + 5
        self.assertEqual(z.value, 7)
        z.backward()
        self.assertEqual(x.grad, 1)

        x = np.random.rand(5, 4)
        y = np.random.rand(4)
        p = ag.Parameter(y)
        z = x + p
        self.assertTrue((z.value == x + y).all())
        z.backward(np.ones((5, 4)))
        self.assertTrue((p.grad == np.ones(4) * 5).all())


if __name__ == '__main__':
    unittest.main()

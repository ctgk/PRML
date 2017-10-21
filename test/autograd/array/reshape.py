import unittest
import numpy as np
import prml.autograd as ag


class TestReshape(unittest.TestCase):

    def test_reshape(self):
        self.assertRaises(ValueError, ag.reshape, 1, (2, 3))

        x = np.random.rand(2, 6)
        p = ag.Parameter(x)
        y = p.reshape(3, 4)
        self.assertTrue((x.reshape(3, 4) == y.value).all())
        y.backward(np.ones((3, 4)))
        self.assertTrue((p.grad == np.ones((2, 6))).all())


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import prml.autograd as ag


class TestFlatten(unittest.TestCase):

    def test_flatten(self):
        self.assertRaises(TypeError, ag.flatten, "abc")
        self.assertRaises(ValueError, ag.flatten, np.ones(1))

        x = np.random.rand(5, 4)
        p = ag.Parameter(x)
        y = p.flatten()
        self.assertTrue((y.value == x.flatten()).all())
        y.backward(np.ones(20))
        self.assertTrue((p.grad == np.ones((5, 4))).all())


if __name__ == '__main__':
    unittest.main()

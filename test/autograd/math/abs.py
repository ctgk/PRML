import unittest
import numpy as np
import prml.autograd as ag


class TestAbs(unittest.TestCase):

    def test_abs(self):
        np.random.seed(1234)
        x = ag.Parameter(np.random.randn(5, 7))
        sign = np.sign(x.value)
        y = ag.abs(x)
        self.assertTrue((y.value == np.abs(x.value)).all())

        for _ in range(10000):
            x.cleargrad()
            y = ag.abs(x)
            ag.square(y - 0.01).sum().backward()
            x.value -= x.grad * 0.001
        self.assertTrue(np.allclose(x.value, 0.01 * sign))


if __name__ == '__main__':
    unittest.main()

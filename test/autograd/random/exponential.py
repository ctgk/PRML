import unittest
import numpy as np
import prml.autograd as ag


class TestExponential(unittest.TestCase):

    def test_exponential(self):
        np.random.seed(1234)
        obs = np.random.gamma(1, 1 / 0.5, size=1000)
        a = ag.Parameter(0)
        for _ in range(100):
            a.cleargrad()
            x = ag.random.Exponential(ag.softplus(a), data=obs)
            x.log_pdf().sum().backward()
            a.value += a.grad * 0.001
        self.assertAlmostEqual(x.rate.value, 0.475135117)


if __name__ == '__main__':
    unittest.main()

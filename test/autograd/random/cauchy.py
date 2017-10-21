import unittest
import numpy as np
import prml.autograd as ag


class TestCauchy(unittest.TestCase):

    def test_cauchy(self):
        np.random.seed(1234)
        obs = np.random.standard_cauchy(size=10000)
        obs = 2 * obs + 1
        loc = ag.Parameter(0)
        s = ag.Parameter(1)
        for _ in range(100):
            loc.cleargrad()
            s.cleargrad()
            x = ag.random.Cauchy(loc, ag.softplus(s), data=obs)
            x.log_pdf().sum().backward()
            loc.value += loc.grad * 0.001
            s.value += s.grad * 0.001
        self.assertAlmostEqual(x.loc.value, 1, places=1)
        self.assertAlmostEqual(x.scale.value, 2, places=1)


if __name__ == '__main__':
    unittest.main()

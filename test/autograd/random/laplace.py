import unittest
import numpy as np
import prml.autograd as ag


class TestLaplace(unittest.TestCase):

    def test_laplace(self):
        obs = np.arange(3)
        loc = ag.Parameter(0)
        s = ag.Parameter(1)
        for _ in range(1000):
            loc.cleargrad()
            s.cleargrad()
            x = ag.random.Laplace(loc, ag.softplus(s), data=obs)
            x.log_pdf().sum().backward()
            loc.value += loc.grad * 0.01
            s.value += s.grad * 0.01
        self.assertAlmostEqual(x.loc.value, np.median(obs), places=1)
        self.assertAlmostEqual(x.scale.value, np.mean(np.abs(obs - x.loc.value)), places=1)


if __name__ == '__main__':
    unittest.main()

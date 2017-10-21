import unittest
import prml.autograd as ag


class TestGamma(unittest.TestCase):

    def test_gamma(self):
        self.assertEqual(24, ag.gamma(5).value)
        a = ag.Parameter(2.5)
        eps = 1e-5
        b = ag.gamma(a)
        b.backward()
        num_grad = ((ag.gamma(a + eps) - ag.gamma(a - eps)) / (2 * eps)).value
        self.assertAlmostEqual(a.grad, num_grad)


if __name__ == '__main__':
    unittest.main()

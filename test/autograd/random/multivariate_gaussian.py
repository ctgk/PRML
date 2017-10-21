import unittest
import numpy as np
import prml.autograd as ag


class TestMultivariateGaussian(unittest.TestCase):

    def test_multivariate_gaussian(self):
        self.assertRaises(ValueError, ag.random.MultivariateGaussian, np.zeros(2), np.eye(3))
        self.assertRaises(ValueError, ag.random.MultivariateGaussian, np.zeros(2), np.eye(2) * -1)

        x_train = np.array([
            [1., 1.],
            [1., -1],
            [-1., 1.],
            [-1., -2.]
        ])
        mu = ag.Parameter(np.ones(2))
        cov = ag.Parameter(np.eye(2) * 2)
        for _ in range(1000):
            mu.cleargrad()
            cov.cleargrad()
            x = ag.random.MultivariateGaussian(mu, cov + cov.transpose(), data=x_train)
            log_likelihood = x.log_pdf().sum()
            log_likelihood.backward()
            mu.value += 0.1 * mu.grad
            cov.value += 0.1 * cov.grad
        self.assertTrue(np.allclose(mu.value, x_train.mean(axis=0)))
        self.assertTrue(np.allclose(np.cov(x_train, rowvar=False, bias=True), x.cov.value))


if __name__ == '__main__':
    unittest.main()

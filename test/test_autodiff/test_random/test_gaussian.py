import unittest

import numpy as np

from prml import autodiff


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):
        samples = autodiff.random.gaussian(-2.3, 0.5, size=100000).value
        self.assertAlmostEqual(-2.3, np.mean(samples), places=2)
        self.assertAlmostEqual(0.5, np.std(samples), places=2)

    def test_multivariate_gaussian(self):
        expect_mean = np.array([1, -1.5])
        expect_covariance = np.array([[0.2, 0.1], [0.1, 0.2]])
        samples = autodiff.random.multivariate_gaussian(
            expect_mean, expect_covariance, size=(100000, 2)).value
        actual_mean = np.mean(samples, axis=0)
        actual_covariance = np.cov(samples, rowvar=False)
        self.assertTrue(
            np.allclose(expect_mean, actual_mean, atol=1e-2),
            msg=f"{expect_mean}\n{actual_mean}")
        self.assertTrue(
            np.allclose(expect_covariance, actual_covariance, atol=1e-2),
            msg=f"{expect_covariance}\n{actual_covariance}")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

from prml import autodiff


class TestExponential(unittest.TestCase):

    def setUp(self):
        self.previous_dtype = autodiff.config.dtype
        autodiff.config.dtype = np.float64

    def tearDown(self):
        autodiff.config.dtype = self.previous_dtype

    def test_exponential_forward(self):
        rate = np.random.uniform(0.5, 100)
        sample_expect = np.random.exponential(1 / rate, size=100000)
        sample_actual = autodiff.random.exponential(rate, size=100000)
        self.assertAlmostEqual(
            sample_expect.mean(), sample_actual.value.mean(), 1)
        self.assertAlmostEqual(
            sample_expect.std(), sample_actual.value.std(), 1)

    def test_exponential_backward(self):
        stdexp = np.random.standard_exponential()

        def standard_exponential(*args, **kwargs):
            return stdexp

        rate = np.random.uniform(0.5, 100)
        with patch("numpy.random.standard_exponential", standard_exponential):
            expect = autodiff.numerical_gradient(
                autodiff.random.exponential, 1e-8, rate)
            rate = autodiff.asarray(rate)
            autodiff.random.exponential(rate).backprop()
            actual = rate.grad
            self.assertAlmostEqual(expect[0].value[0], actual[0])

    def test_exponential_logpdf_forward(self):
        rate = np.random.uniform(0.1, 10)
        x = np.random.uniform(0.1, 10)

        def exponential_logpdf(x, rate):
            return np.log(rate * np.exp(-rate * x))

        expect = exponential_logpdf(x, rate)
        actual = autodiff.random.exponential_logpdf(x, rate).value[0]
        self.assertAlmostEqual(expect, actual)

    def test_exponential_logpdf_backward(self):
        rate = autodiff.asarray(np.random.uniform(0.1, 100))
        x = autodiff.asarray(np.random.uniform(0.1, 100))

        expect_dx, expect_drate = autodiff.numerical_gradient(
            autodiff.random.exponential_logpdf, 1e-8, x, rate)
        autodiff.random.exponential_logpdf(x, rate).backprop()
        actual_dx, actual_drate = x.grad[0], rate.grad[0]
        self.assertAlmostEqual(expect_dx.value[0], actual_dx, 3)
        self.assertAlmostEqual(expect_drate.value[0], actual_drate, 3)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from prml import autodiff


class TestBernoulli(unittest.TestCase):

    def test_bernoulli_raise(self):
        with self.assertRaises(ValueError):
            autodiff.random.bernoulli(0.1, -0.5)
        with self.assertRaises(ValueError):
            autodiff.random.bernoulli(temperature=0.1, size=100)

    def test_bernoulli(self):
        expect = 0.9
        logit = autodiff.logit(expect)
        actual_sample = autodiff.random.bernoulli(
            logit=logit, temperature=0.1, size=100000
        ).value
        actual = np.mean(actual_sample)
        self.assertAlmostEqual(expect, actual, places=2)


if __name__ == "__main__":
    unittest.main()

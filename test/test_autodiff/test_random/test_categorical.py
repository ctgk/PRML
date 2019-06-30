import unittest

import numpy as np

from prml import autodiff


class TestCategorical(unittest.TestCase):

    def test_categorical_raise(self):
        with self.assertRaises(ValueError):
            autodiff.random.categorical([0.2, 0.3, 0.5], [-1, 2, 5])
        with self.assertRaises(ValueError):
            autodiff.random.categorical(temperature=10, size=1)

    def test_categorical(self):
        expect = np.array([0.2, 0.3, 0.5])
        logit = autodiff.log(expect)
        actual_sample = autodiff.random.categorical(
            logit=logit, temperature=0.1, size=(100000, 3)
        ).value
        actual = np.mean(actual_sample, axis=0)
        self.assertTrue(np.allclose(expect, actual, atol=1e-2))


if __name__ == "__main__":
    unittest.main()

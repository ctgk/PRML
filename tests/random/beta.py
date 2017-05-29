import unittest
import numpy as np
from prml.random import Beta


class TestBeta(unittest.TestCase):

    def test_init(self):
        beta = Beta(np.zeros(2), np.ones(2))
        self.assertTrue(np.allclose(beta.n_ones, np.zeros(2)))
        self.assertTrue(np.allclose(beta.n_zeros, np.ones(2)))

    def test_repr(self):
        beta = Beta()
        self.assertEqual(repr(beta), "Beta(n_ones=[ 1.], n_zeros=[ 1.])")

    def test_mean(self):
        beta = Beta(np.ones(2) * 3, np.ones(2))
        self.assertTrue(np.allclose(beta.mean, 0.75))

    def test_var(self):
        beta = Beta(np.ones(2) * 3, np.ones(2))
        self.assertTrue(np.allclose(beta.var, np.eye(2) * 3 / 80))

    def test_pdf(self):
        beta = Beta(np.ones(2), np.ones(2))
        self.assertTrue(
            np.allclose(beta.pdf(np.random.uniform(size=(5, 2))), 1.)
        )

    def test_draw(self):
        beta = Beta(np.array([3., 1.]), np.array([1., 3.]))
        self.assertEqual(beta.draw().shape, (1, 2))
        sample = beta.draw(1000)
        self.assertTrue(
            np.allclose(np.mean(sample, axis=0), [0.75, 0.25], 1e-1, 1e-1)
        )

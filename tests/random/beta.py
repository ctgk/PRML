import unittest
import numpy as np
from prml.random import Beta


class TestBeta(unittest.TestCase):

    def test_init(self):
        beta = Beta(1, 1)
        self.assertEqual(beta.n_ones, 1)
        self.assertEqual(beta.n_zeros, 1)
        beta = Beta(np.zeros(2), np.ones(2))
        self.assertTrue(np.allclose(beta.n_ones, np.zeros(2)))
        self.assertTrue(np.allclose(beta.n_zeros, np.ones(2)))

    def test_repr(self):
        beta = Beta(n_ones=np.ones(1), n_zeros=np.ones(1))
        self.assertEqual(repr(beta), "Beta(\nn_ones=[ 1.],\nn_zeros=[ 1.])")

    def test_mean(self):
        beta = Beta(np.ones(2) * 3, np.ones(2))
        self.assertTrue(np.allclose(beta.mean, 0.75))

    def test_var(self):
        beta = Beta(np.ones(2) * 3, np.ones(2))
        self.assertTrue(np.allclose(beta.var, np.ones(2) * 3 / 80))

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
        beta = Beta(
            np.random.randint(1, 9, size=(3, 4)),
            np.random.randint(1, 9, size=(3, 4))
        )
        sample = beta.draw(1000)
        self.assertEqual(sample.shape, (1000, 3, 4))

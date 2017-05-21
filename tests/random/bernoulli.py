import unittest
import numpy as np
from prml.random import Bernoulli, Beta


class TestBernoulli(unittest.TestCase):

    def test_init(self):
        b = Bernoulli()
        self.assertTrue(b.mu is None)
        b = Bernoulli(mu=np.ones(3))
        self.assertTrue(b.ndim == 3)
        self.assertTrue(np.allclose(b.mu, np.ones(3)))

    def test_repr(self):
        b = Bernoulli(mu=np.zeros(5))
        self.assertEqual(repr(b), "Bernoulli(mu=[ 0.  0.  0.  0.  0.])")

    def test_mean(self):
        b = Bernoulli(np.ones(3))
        self.assertTrue(np.allclose(b.mean, 1))

    def test_var(self):
        b = Bernoulli(np.ones(3) * 0.5)
        self.assertTrue(np.allclose(b.var, np.eye(3) * 0.25))

    def test_ml(self):
        b = Bernoulli()
        b.ml(np.ones((4, 5)))
        self.assertTrue(b.ndim == 5)
        self.assertTrue(np.allclose(b.mu, 1))

    def test_map(self):
        mu = Beta(n_ones=np.ones(1), n_zeros=np.ones(1))
        model = Bernoulli(mu=mu)
        model.map(np.array([1., 1., 0.])[:, None])
        self.assertTrue((model.mu == 2 / 3))

    def test_bayes(self):
        mu = Beta(n_ones=np.ones(1), n_zeros=np.ones(1))
        model = Bernoulli(mu=mu)
        self.assertEqual(
            repr(model),
            "Bernoulli(mu=Beta(n_ones=[ 1.], n_zeros=[ 1.]))"
        )
        model.bayes(np.array([1., 1., 0.])[:, None])
        self.assertEqual(
            repr(model),
            "Bernoulli(mu=Beta(n_ones=[ 3.], n_zeros=[ 2.]))"
        )

    def test_proba(self):
        b = Bernoulli(np.ones(2))
        self.assertTrue(
            np.allclose(b.proba(np.ones((3, 2))), 1)
        )
        self.assertTrue(
            np.allclose(b.proba(np.zeros((4, 2))), 0)
        )

    def test_draw(self):
        b = Bernoulli(np.ones(6))
        self.assertTrue(
            np.allclose(b.draw(), np.ones((1, 6)))
        )
        b = Bernoulli(np.zeros(2))
        self.assertTrue(
            np.allclose(b.draw(4), np.zeros((4, 2)))
        )

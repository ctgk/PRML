import unittest
import numpy as np
from prml.random import Bernoulli, Beta


class TestBernoulli(unittest.TestCase):

    def test_init(self):
        b = Bernoulli()
        self.assertTrue(b.prob is None)
        b = Bernoulli(prob=np.ones(3))
        self.assertTrue(b.ndim == 1)
        self.assertTrue(b.size == 3)
        self.assertTrue(b.shape == (3,))
        self.assertTrue(np.allclose(b.prob, np.ones(3)))
        b = Bernoulli(prob=0.5)
        self.assertEqual(b.ndim, 0)
        self.assertEqual(b.size, 1)
        self.assertEqual(b.shape, ())
        self.assertEqual(b.prob, 0.5)

    def test_repr(self):
        b = Bernoulli()
        self.assertEqual(repr(b), "Bernoulli(prob=None)")
        b = Bernoulli(prob=np.zeros(5))
        self.assertEqual(repr(b), "Bernoulli(prob=[ 0.  0.  0.  0.  0.])")

    def test_mean(self):
        b = Bernoulli()
        self.assertEqual(b.mean, None)
        b = Bernoulli(0.75)
        self.assertEqual(b.mean, 0.75)

    def test_var(self):
        b = Bernoulli()
        self.assertEqual(b.var, None)
        b = Bernoulli(np.ones(3) * 0.5)
        self.assertTrue(np.allclose(b.var, np.ones(3) * 0.25))

    def test_ml(self):
        b = Bernoulli()
        b.ml(np.array([0., 1., 1., 1.]))
        self.assertEqual(b.prob, 0.75)
        b = Bernoulli()
        b.ml(np.ones((4, 5, 2)))
        self.assertTrue(b.shape == (5, 2))
        self.assertTrue(np.allclose(b.prob, np.ones((5, 2))))

    def test_map(self):
        mu = Beta(n_ones=np.ones(1), n_zeros=np.ones(1))
        model = Bernoulli(prob=mu)
        model.map(np.array([1., 1., 0.])[:, None])
        self.assertTrue((model.prob == 2 / 3))

    def test_bayes(self):
        mu = Beta(n_ones=np.ones(1), n_zeros=np.ones(1))
        model = Bernoulli(prob=mu)
        self.assertEqual(
            repr(model),
            "Bernoulli(prob=\nBeta(\nn_ones=[ 1.],\nn_zeros=[ 1.])\n)"
        )
        model.bayes(np.array([1., 1., 0.])[:, None])
        self.assertEqual(
            repr(model),
            "Bernoulli(prob=\nBeta(\nn_ones=[ 3.],\nn_zeros=[ 2.])\n)"
        )

    def test_pdf(self):
        b = Bernoulli(np.ones(2))
        self.assertTrue(
            np.allclose(b.pdf(np.ones((3, 2))), 1)
        )
        self.assertTrue(
            np.allclose(b.pdf(np.zeros((4, 2))), 0)
        )

    def test_draw(self):
        b = Bernoulli(np.ones((6, 2)))
        self.assertTrue(
            np.allclose(b.draw(), np.ones((1, 6, 2)))
        )
        b = Bernoulli(np.zeros((2, 3)))
        self.assertTrue(
            np.allclose(b.draw(4), np.zeros((4, 2, 3)))
        )

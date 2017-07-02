import unittest
import numpy as np
from prml.random import MultivariateGaussian


class TestMultivariateGaussian(unittest.TestCase):

    def test_init(self):
        g = MultivariateGaussian(mean=np.ones(2))
        self.assertTrue((g.mean == 1).all())
        self.assertTrue(g.var is None)
        g = MultivariateGaussian(mean=np.ones(3), var=np.eye(3) * 2)
        self.assertTrue(np.allclose(g.precision, np.eye(3) * 0.5))

    def test_repr(self):
        g = MultivariateGaussian(mean=np.ones(2), var=np.eye(2) * 2)
        self.assertEqual(
            repr(g),
            "MultivariateGaussian(\nmean=[ 1.  1.],\nvar=\n[[ 2.  0.]\n [ 0.  2.]]\n)"
        )

    def test_mean(self):
        g = MultivariateGaussian(mean=np.ones(3))
        self.assertTrue((g.mean == 1).all())

    def test_var(self):
        g = MultivariateGaussian(mean=np.ones(2), var=np.eye(2) * 2)
        self.assertTrue((g.var == np.eye(2) * 2).all())

    def test_ml(self):
        g = MultivariateGaussian()
        g.ml(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean == 0).all())
        self.assertTrue((g.var == 0.5 * np.eye(2)).all())

    def test_map(self):
        g = MultivariateGaussian(mean=MultivariateGaussian(np.zeros(2), np.eye(2)), var=np.eye(2))
        g.map(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean == 0).all())
        g1 = MultivariateGaussian(mean=MultivariateGaussian(np.zeros(2), np.eye(2)), var=np.eye(2))
        g1.bayes(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean == g1.mean.mean).all())

    def test_bayes(self):
        g = MultivariateGaussian(mean=MultivariateGaussian(np.zeros(2), np.eye(2)), var=np.eye(2))
        g.bayes(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean.var == np.eye(2) * 0.2).all())

    def test_pdf(self):
        g = MultivariateGaussian(mean=np.zeros(1), var=np.ones((1, 1)))
        self.assertEqual(g.pdf(np.zeros((1, 1))), (2 * np.pi) ** -0.5)
        self.assertEqual(g.pdf(np.ones((1, 1))), (2 * np.pi) ** -0.5 * np.exp(-0.5))

    def test_draw(self):
        g = MultivariateGaussian(mean=np.ones(2), var=np.eye(2) * 2)
        sample = g.draw(10000)
        self.assertTrue(np.allclose(np.mean(sample, 0), g.mean, 1e-1, 1e-1))
        self.assertTrue(np.allclose(np.cov(sample.T), g.var, 1e-1, 1e-1))

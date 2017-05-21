import unittest
import numpy as np
from prml.random import Gaussian


class TestGaussian(unittest.TestCase):

    def test_init(self):
        g = Gaussian(mu=np.ones(2))
        self.assertTrue((g.mean == 1).all())
        self.assertTrue(g.var is None)
        g = Gaussian(mu=np.ones(3), var=np.eye(3) * 2)
        self.assertTrue(np.allclose(g.precision, np.eye(3) * 0.5))

    def test_repr(self):
        g = Gaussian(mu=np.ones(2), var=np.eye(2) * 2)
        self.assertEqual(
            repr(g),
            "Gaussian(\nmu=[ 1.  1.],\nvar=\n[[ 2.  0.]\n [ 0.  2.]]\n)"
        )

    def test_mean(self):
        g = Gaussian(mu=np.ones(3))
        self.assertTrue((g.mu == 1).all())

    def test_var(self):
        g = Gaussian(mu=np.ones(2), var=np.eye(2) * 2)
        self.assertTrue((g.var == np.eye(2) * 2).all())

    def test_ml(self):
        g = Gaussian()
        g.ml(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mu == 0).all())
        self.assertTrue((g.var == 0.5 * np.eye(2)).all())

    def test_map(self):
        g = Gaussian(mu=Gaussian(np.zeros(2), np.eye(2)), var=np.eye(2))
        g.map(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mu == 0).all())
        g1 = Gaussian(mu=Gaussian(np.zeros(2), np.eye(2)), var=np.eye(2))
        g1.bayes(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mu == g1.mu.mu).all())

    def test_bayes(self):
        g = Gaussian(mu=Gaussian(np.zeros(2), np.eye(2)), var=np.eye(2))
        g.bayes(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mu.var == np.eye(2) * 0.2).all())

    def test_proba(self):
        g = Gaussian(mu=np.zeros(1), var=np.ones((1, 1)))
        self.assertEqual(g.proba(np.zeros((1, 1))), (2 * np.pi) ** -0.5)
        self.assertEqual(g.proba(np.ones((1, 1))), (2 * np.pi) ** -0.5 * np.exp(-0.5))

    def test_draw(self):
        g = Gaussian(mu=np.ones(2), var=np.eye(2) * 2)
        sample = g.draw(10000)
        self.assertTrue(np.allclose(np.mean(sample, 0), g.mean, 1e-1, 1e-1))
        self.assertTrue(np.allclose(np.cov(sample, rowvar=False), g.var, 1e-1, 1e-1))

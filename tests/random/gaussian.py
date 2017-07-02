import unittest
import numpy as np
from prml.random import Gaussian


class TestGaussian(unittest.TestCase):

    def test_init(self):
        g = Gaussian(mean=np.ones(2))
        self.assertTrue((g.mean == 1).all())
        self.assertTrue(g.var is None)
        g = Gaussian(mean=np.ones(3), var=np.ones(3) * 2)
        self.assertTrue(np.allclose(g.precision, np.ones(3) * 0.5))

    def test_repr(self):
        g = Gaussian(mean=np.ones(2), var=np.ones(2) * 2)
        self.assertEqual(
            repr(g),
            "Gaussian(\nmean=\n[ 1.  1.],\nvar=\n[ 2.  2.]\n)"
        )

    def test_mean(self):
        g = Gaussian(mean=np.ones(3))
        self.assertTrue((g.mean == 1).all())

    def test_var(self):
        g = Gaussian(mean=np.ones((2, 3)), var=np.ones((2, 3)) * 2)
        self.assertTrue((g.var == np.ones((2, 3)) * 2).all())

    def test_ml(self):
        g = Gaussian()
        g.ml(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean == 0).all())
        self.assertTrue((g.var == 0.5 * np.ones(2)).all())

    def test_map(self):
        g = Gaussian(mean=Gaussian(np.zeros(2), np.ones(2)), var=np.ones(2))
        g.map(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean == 0).all())
        g1 = Gaussian(mean=Gaussian(np.zeros(2), np.ones(2)), var=np.ones(2))
        g1.bayes(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean == g1.mean.mean).all())

    def test_bayes(self):
        g = Gaussian(mean=Gaussian(np.zeros(2), np.ones(2)), var=np.ones(2))
        g.bayes(np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        self.assertTrue((g.mean.var == np.ones(2) * 0.2).all())

    def test_pdf(self):
        g = Gaussian(mean=np.zeros(1), var=np.ones(1))
        self.assertEqual(g.pdf(np.zeros((1, 1))), (2 * np.pi) ** -0.5)
        self.assertEqual(g.pdf(np.ones((1, 1))), (2 * np.pi) ** -0.5 * np.exp(-0.5))

    def test_draw(self):
        g = Gaussian(mean=np.ones((2, 4)), var=np.ones((2, 4)) * 2)
        sample = g.draw(10000)
        self.assertTrue(np.allclose(np.mean(sample, 0), g.mean, 1e-1, 1e-1))
        self.assertTrue(np.allclose(np.var(sample, 0), g.var, 1e-1, 1e-1))

import unittest
import numpy as np
from prml.random import Dirichlet


class TestDirichlet(unittest.TestCase):

    def test_init(self):
        d = Dirichlet(np.ones(3))
        self.assertTrue((d.concentration == 1).all())
        self.assertEqual(d.size, 3)
        self.assertEqual(d.ndim, 1)
        self.assertEqual(d.shape, (3,))

    def test_repr(self):
        d = Dirichlet(np.ones(3))
        self.assertEqual(repr(d), "Dirichlet(concentration=[ 1.  1.  1.])")

    def test_mean(self):
        d = Dirichlet(np.ones(4))
        self.assertTrue((d.mean == 0.25).all())

    def test_var(self):
        d = Dirichlet(np.array([1, 2, 3]))
        var = np.array(
            [[5, -2, -3],
             [-2, 8, -6],
             [-3, -6, 9]]
        ) / 36 / 7
        self.assertTrue(np.allclose(var, d.var))

    def test_pdf(self):
        d = Dirichlet(np.ones(4))
        self.assertTrue((d.pdf(np.random.uniform(size=(5, 4))) == 6).all())

    def test_draw(self):
        d = Dirichlet(np.array([1., 2., 1.]))
        self.assertEqual(d.draw().shape, (1, 3))
        sample = d.draw(1000)
        self.assertTrue(
            np.allclose(np.mean(sample, axis=0), [0.25, 0.5, 0.25], 1e-1, 1e-1)
        )

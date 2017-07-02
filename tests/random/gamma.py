import unittest
import numpy as np
from prml.random import Gamma


class TestGamma(unittest.TestCase):

    def test_init(self):
        g = Gamma(shape=4, rate=6)
        self.assertEqual(g.shape_, 4)
        self.assertEqual(g.rate, 6)

    def test_repr(self):
        g = Gamma(shape=1, rate=1)
        self.assertEqual(repr(g), "Gamma(\nshape=\n1,\nrate=\n1\n)")

    def test_mul(self):
        g = Gamma(shape=1, rate=2)
        h = g * 2
        self.assertEqual(repr(g), "Gamma(\nshape=\n1,\nrate=\n2\n)")
        self.assertEqual(repr(h), "Gamma(\nshape=\n1,\nrate=\n1.0\n)")
        h = 2 * h
        self.assertEqual(repr(h), "Gamma(\nshape=\n1,\nrate=\n0.5\n)")
        h *= 0.5
        self.assertEqual(repr(h), "Gamma(\nshape=\n1,\nrate=\n1.0\n)")

    def test_div(self):
        g = Gamma(shape=1, rate=2)
        h = g / 2
        self.assertEqual(repr(g), "Gamma(\nshape=\n1,\nrate=\n2\n)")
        self.assertEqual(repr(h), "Gamma(\nshape=\n1,\nrate=\n4\n)")
        h /= 2
        self.assertEqual(repr(h), "Gamma(\nshape=\n1,\nrate=\n8\n)")

    def test_mean(self):
        g = Gamma(shape=4, rate=6)
        self.assertEqual(g.mean, 2 / 3)

    def test_var(self):
        g = Gamma(shape=4, rate=6)
        self.assertEqual(g.var, 1 / 9)

    def test_pdf(self):
        g = Gamma(shape=4, rate=6)
        self.assertTrue(np.allclose(g.pdf(np.ones(1)), 0.53541047))

    def test_draw(self):
        g = Gamma(shape=4, rate=6)
        self.assertTrue(np.allclose(g.draw(10000).mean(axis=0), 2 / 3, 1e-1, 1e-1))

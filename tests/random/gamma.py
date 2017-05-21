import unittest
import numpy as np
from prml.random import Gamma


class TestGamma(unittest.TestCase):

    def test_init(self):
        g = Gamma(a=4, b=6)
        self.assertEqual(g.a, 4)
        self.assertEqual(g.b, 6)

    def test_repr(self):
        g = Gamma(a=1, b=1)
        self.assertEqual(repr(g), "Gamma(a=1, b=1)")

    def test_mul(self):
        g = Gamma(a=1, b=2)
        h = g * 2
        self.assertEqual(repr(g), "Gamma(a=1, b=2)")
        self.assertEqual(repr(h), "Gamma(a=1, b=1.0)")
        h = 2 * h
        self.assertEqual(repr(h), "Gamma(a=1, b=0.5)")
        h *= 0.5
        self.assertEqual(repr(h), "Gamma(a=1, b=1.0)")

    def test_div(self):
        g = Gamma(a=1, b=2)
        h = g / 2
        self.assertEqual(repr(g), "Gamma(a=1, b=2)")
        self.assertEqual(repr(h), "Gamma(a=1, b=4)")
        h /= 2
        self.assertEqual(repr(h), "Gamma(a=1, b=8)")

    def test_mean(self):
        g = Gamma(a=4, b=6)
        self.assertEqual(g.mean, 2 / 3)

    def test_var(self):
        g = Gamma(a=4, b=6)
        self.assertEqual(g.var, 1 / 9)

    def test_proba(self):
        g = Gamma(a=4, b=6)
        self.assertTrue(np.allclose(g.proba(np.ones((1, 1))), 0.53541047))

    def test_draw(self):
        g = Gamma(a=4, b=6)
        self.assertTrue(np.allclose(g.draw(10000).mean(axis=0), 2 / 3, 1e-1, 1e-1))
